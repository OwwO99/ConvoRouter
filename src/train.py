import os
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger
from openai import OpenAI
from itertools import count
from datetime import datetime
from FlagEmbedding import BGEM3FlagModel
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from env import Env
from utils import *
from task import create_task
from model import ConvoRouter


def setup_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_next_state(state, retriever: BGEM3FlagModel, train_loader, train_emb):
    dial = ''
    for turn in state[::-1]:
        dial += '%s: %s ' % (turn['role'], turn['content'])
    emb = retriever.encode([dial], batch_size=12, max_length=8192)['dense_vecs']
    similarity = emb @ train_emb.T
    ind = [i for i, _ in sorted(enumerate(similarity[0]), key=lambda x: x[1], reverse=True)][0]
    next_state = state.copy()
    next_state.extend(train_loader.dataset.data[ind]['final_state'][len(train_loader.dataset.data[ind]['state']): len(train_loader.dataset.data[ind]['state']) + 2])
    return next_state


def evaluate(testenv: Env, router: ConvoRouter, retriever, train_loader, train_emb, tid, epoch, use_realization=False, system_set=None, task=None):
    AT = 0
    SR = 0
    router.eval()
    with torch.no_grad():
        for test_id in tqdm(range(len(testenv.task.dataset))):
            logger.info('================test tuple:{}===================='.format(test_id))
            realizations = None
            if use_realization and os.path.exists(f'data/{system_set}/realization/{task}/dia-{test_id}.pkl'):
                realizations = read_pkl(f'data/{system_set}/realization/{task}/dia-{test_id}.pkl')
            state = testenv.reset(test_id, realizations=realizations)
            for t in count():
                next_state = get_next_state(state.system_messages[1:], retriever=retriever, train_loader=train_loader, train_emb=train_emb)
                action = router.select_action(state.system_messages[1:], next_state=next_state)
                state, checkscore, done = testenv.step(action.item())
                if done:
                    break
            
            conversation = testenv.conversation_all
            AT += t + 1
            SR += checkscore / len(testenv.state.case['checklist'])
            dict2json(conversation, f'results/{tid}/test/{epoch + 1}-{test_id}.json')
            testenv.save_dia(f'results/{tid}/realization/dia-{epoch + 1}-{test_id}.pkl')
        logger.info(f'Test Epoch {epoch + 1}. AT: {AT / len(testenv.task.dataset)}, SR: {SR / len(testenv.task.dataset)}')
        torch.save(router.state_dict(), f'results/{tid}/checkpoints/{epoch + 1}.pth')


def train(testenv: Env, router: ConvoRouter, train_loader, num_epochs=15, lr=1e-6, retriever=None, train_emb=None, tid=None, use_realization=False, system_set=None, task=None):
    optimizer = AdamW(router.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    for epoch in range(num_epochs):
        router.train()
        total_loss = 0
        for state, next_state, reward in tqdm(train_loader):
            optimizer.zero_grad()
            logits = router(state, next_state)
            targets = reward.argmax(dim=-1)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {total_loss / len(train_loader)}")
        evaluate(testenv=testenv, router=router, retriever=retriever, train_loader=train_loader, train_emb=train_emb, tid=tid, epoch=epoch, use_realization=use_realization, system_set=system_set, task=task)
        scheduler.step()


def collate_fn(batch, device="cuda"):
    input_ids = torch.stack([item[0]["input_ids"].squeeze(0) for item in batch])
    attention_masks = torch.stack([item[0]["attention_mask"].squeeze(0) for item in batch])
    next_input_ids = torch.stack([item[1]["input_ids"].squeeze(0) for item in batch])
    next_attention_masks = torch.stack([item[1]["attention_mask"].squeeze(0) for item in batch])
    rewards = torch.stack([item[2] for item in batch])

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_masks.to(device),
    }, {
        "input_ids": next_input_ids.to(device),
        "attention_mask": next_attention_masks.to(device),
    }, rewards.to(device)


class SearchData(Dataset):
    def __init__(self, task, system_set, tokenizer, device, max_seq_length=1408):
        super().__init__()
        self.data = json2dict(f'data/{system_set}/{task}_train.json')
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.device = device

    def build_input(self, state):
        dial_id = []
        for turn in state[::-1]:
            s = self.tokenizer.encode("%s: %s" % (turn['role'], turn['content']))
            if len(dial_id) + len(s) > self.max_seq_length:
                break
            dial_id = s[1:] + dial_id
        inp = s[:1] + dial_id
        
        seq_length = len(inp)
        attention_mask = [1] * seq_length

        if seq_length < self.max_seq_length:
            pad_len = self.max_seq_length - seq_length
            inp = inp + [0] * pad_len
            attention_mask = attention_mask + [0] * pad_len
        else:
            inp = inp[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
        
        return {
            "input_ids": torch.tensor([inp]).long().to(self.device),
            "attention_mask": torch.tensor([attention_mask]).to(self.device),
        }
    
    def __getitem__(self, index):
        state = self.build_input(self.data[index]['state'])
        next_state = self.build_input(self.data[index]['r_state'])
        reward = torch.tensor(self.data[index]['reward'], dtype=torch.float32).to(self.device)
                
        return state, next_state, reward
    
    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--user_type', type=str, default='API_NAME')
    args.add_argument('--eval_type', type=str, default='API_NAME')
    args.add_argument('--user_model', type=str, default='user_model')
    args.add_argument('--eval_model', type=str, default='eval_model')
    args.add_argument('--system_set', type=str, default='LLM_Pool_Name')
    args.add_argument("--temperature", type=float, default=0.0)
    args.add_argument('--max_turn', type=int, default=8)
    args.add_argument('--max_new_tokens', type=int, default=512)
    args.add_argument('--task', type=str, default='sharegpt', choices=['meddg', 'jddc', 'sharegpt'])
    args.add_argument('--encoder_path', type=str, default='Qwen/Qwen2.5-0.5B')
    args.add_argument('--learning_rate', type=float, default=1e-5)
    args.add_argument("--max_seq_length", type=int, default=1408)
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--num_epochs', type=int, default=15)
    args = args.parse_args()
    logger.info(vars(args))


    setup_seed(args.seed)
    tid = datetime.now().strftime("%Y%m%d%H%M%S")
    tid = f'{args.task}-{tid}'
    os.makedirs('results', exist_ok=True)
    os.makedirs(f'results/{tid}/test', exist_ok=True)
    os.makedirs(f'results/{tid}/checkpoints', exist_ok=True)
    os.makedirs(f'results/{tid}/realization', exist_ok=True)
    logger.add(f'logs/log-{tid}.log')


    API = {
        "API_NAME": {
            "base_url": "",
            "api_key": ""
        },
    }
    model_list = {
        # Define your own LLM pools here.
        # Each pool can include multiple API providers and corresponding models.
        'LLM_Pool_Name': {
            'type': ['API_NAME1', 'API_NAME2', 'API_NAME3'],
            'model': ['LLM1', 'LLM2', 'LLM3'],
        },
    }
    user_client = OpenAI(
        base_url=API[args.user_type]['base_url'],
        api_key=API[args.user_type]['api_key']
    )
    eval_client = OpenAI(
        base_url=API[args.eval_type]['base_url'],
        api_key=API[args.eval_type]['api_key']
    )
    system_clients = []
    for api_type in model_list[args.system_set]['type']:
        system_clients.append(
            OpenAI(
                base_url=API[api_type]['base_url'],
                api_key=API[api_type]['api_key']
            )
        )
    

    testset = read_json(f'data/{args.task}_test.jsonl')
    testtask = create_task(testset, args.task)
    testenv = Env(
        task=testtask,
        user_client=user_client,
        user_model=args.user_model,
        eval_client=eval_client,
        eval_model=args.eval_model,
        system_clients=system_clients,
        system_models=model_list[args.system_set]['model'],
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_turn=args.max_turn,
    )

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_path)
    encoder_model = AutoModelForCausalLM.from_pretrained(
        args.encoder_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    router = ConvoRouter(
        encoder=encoder_model,
        tokenizer=tokenizer,
        hidden_dim=encoder_model.config.hidden_size,
        num_models=len(model_list[args.system_set]['model']),
        max_seq_length=args.max_seq_length,
        device=device
    )
    retriever = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)
    router.to(device)


    train_dataset = SearchData(task=args.task, system_set=args.system_set, tokenizer=tokenizer, device=device, max_seq_length=args.max_seq_length)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    train_emb = read_pkl(f'data/{args.system_set}/emb/{args.task}_train_emb.pkl')
    train(
        testenv=testenv,
        router=router,
        train_loader=train_loader,
        num_epochs=args.num_epochs,
        lr=args.learning_rate,
        retriever=retriever,
        train_emb=train_emb,
        tid=tid,
        use_realization=False,
        system_set=args.system_set,
        task=args.task
    )

import os
import torch
import random
import argparse
import tiktoken
import numpy as np
from tqdm import tqdm
from loguru import logger
from openai import OpenAI
from itertools import count
from datetime import datetime
from env import Env
from utils import *
from task import create_task
from mcts import MCTS


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


def mcts(env: Env, num_MCTS_sims=10, num_models=2, Q_0=0.25, cuct=1.0, gamma=0.99, max_depth=3, lambda1=0, tid=0):
    AT = 0
    SR = 0
    enc = tiktoken.get_encoding("cl100k_base")
    for case_id in tqdm(range(len(env.task.dataset))):
        prev_C = 0
        checkscore = 0
        done = 0
        logger.info('================tuple:{}===================='.format(case_id))
        state = env.reset(case_id)
        dialog_planner = MCTS(env=env, num_models=num_models, Q_0=Q_0, cuct=cuct, gamma=gamma, max_depth=max_depth, enc=enc, lambda1=lambda1)
        for t in count():
            for _ in tqdm(range(num_MCTS_sims)):
                dialog_planner.search(state, prev_C, checkscore, done)
            mcts_policy = dialog_planner.get_action_prob(state, prev_C, checkscore, done)
            prev_C = checkscore
            action = np.argmax(mcts_policy)
            state, checkscore, done = env.step(action)
            if done:
                break
        conversation = env.conversation_all
        AT += t + 1
        SR += checkscore / len(env.state.case['checklist'])
        env.save_dia(f'results/{tid}/realization/dia-{case_id}.pkl')
        dict2json(conversation, f'results/{tid}/train/{case_id}.json')
        dialog_planner.save_mcts(f'results/{tid}/mcts/{case_id}.pkl')
    logger.info(f'MCTS. AT: {AT / len(env.task.dataset)}, SR: {SR / len(env.task.dataset)}')


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
    args.add_argument("--max_new_tokens", type=int, default=512)
    args.add_argument('--task', type=str, default='sharegpt', choices=['meddg', 'jddc', 'sharegpt'])
    args.add_argument('--num_MCTS_sims', type=int, default=10)
    args.add_argument('--Q_0', type=float, default=0.25)
    args.add_argument('--cuct', type=float, default=2)
    args.add_argument('--max_depth', type=int, default=8)
    args.add_argument('--gamma', type=float, default=0.999)
    args.add_argument('--lambda1', type=float, default=0.0)
    args = args.parse_args()
    logger.info(vars(args))


    setup_seed(args.seed)
    tid = datetime.now().strftime("%Y%m%d%H%M%S")
    tid = f'{args.task}-{tid}'
    os.makedirs('results', exist_ok=True)
    os.makedirs(f'results/{tid}/train', exist_ok=True)
    os.makedirs(f'results/{tid}/realization', exist_ok=True)
    os.makedirs(f'results/{tid}/mcts', exist_ok=True)
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

    
    trainset = read_json(f'data/{args.task}_train.jsonl')
    traintask = create_task(trainset, args.task)
    trainenv = Env(
        task=traintask,
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


    mcts(
        env=trainenv,
        num_MCTS_sims=args.num_MCTS_sims,
        num_models=len(model_list[args.system_set]['model']),
        Q_0=args.Q_0,
        cuct=args.cuct,
        gamma=args.gamma,
        max_depth=args.max_depth,
        lambda1=args.lambda1,
        tid=tid
    )

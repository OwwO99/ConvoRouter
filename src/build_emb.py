import os
import argparse
from FlagEmbedding import BGEM3FlagModel
from utils import *


def get_context(state):
    dial = ''
    for turn in state[::-1]:
        dial += '%s: %s ' % (turn['role'], turn['content'])
    return dial


def rank_index_desc(lst):
    return [i for i, _ in sorted(enumerate(lst), key=lambda x: x[1], reverse=True)]


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--system_set', type=str, default='LLM_Pool_Name')
    args.add_argument('--task', type=str, default='sharegpt', choices=['meddg', 'jddc', 'sharegpt'])
    args = args.parse_args()

    os.makedirs(f'data/{args.system_set}/emb', exist_ok=True)
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)
    train = json2dict(f'data/{args.system_set}/{args.task}_train.json')
    trains = [get_context(s['state']) for s in train]
    train_emb = model.encode(trains, batch_size=12, max_length=8192)['dense_vecs']
    similarity_train = train_emb @ train_emb.T
    save_pkl(train_emb, f'data/{args.system_set}/emb/{args.task}_train_emb.pkl')
    for i in range(len(train)):
        ind = rank_index_desc(similarity_train[i])
        for j in ind:
            if train[j]['case'] != train[i]['case']:
                break
        train[i]['r_state'] = train[i]['state'].copy()
        train[i]['r_state'].extend(train[j]['final_state'][len(train[j]['state']): len(train[j]['state']) + 2])
    dict2json(train, f'data/{args.system_set}/{args.task}_train.json')
    
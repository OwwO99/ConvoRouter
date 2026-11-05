import os
import pickle
import argparse
import numpy as np
from utils import *


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--run_id', type=str, required=True)
    args.add_argument('--system_set', type=str, default='LLM_Pool_Name')
    args.add_argument('--task', type=str, default='sharegpt', choices=['meddg', 'jddc', 'sharegpt'])
    args = args.parse_args()

    dataset = []
    os.makedirs(f'data/{args.system_set}', exist_ok=True)
    trainset = read_json(f'data/{args.task}_train.jsonl')
    for i in range(len(trainset)):
        with open(f'results/{args.run_id}/mcts/{i}.pkl', 'rb') as file:
            mcts = pickle.load(file)
        Q = mcts['Q']
        conversation = json2dict(f'results/{args.run_id}/train/{i}.json')
        data = conversation[:-1]
        states = []
        actions_ = []
        action_states = ['']
        rewards = []
        for t in range(len(data)):
            states.append([])
        for t in range(len(data)):
            for j in range(t, len(data)):
                states[j].append({'role': 'user', 'content': data[t]['user']})
            for j in range(t + 1, len(data)):
                states[j].append({'role': 'assistant', 'content': data[t]['system']})
            actions_.append(data[t]['model_id'])
        actions = actions_[: -1]
        final_state = states[-1] + [
            {'role': 'assistant', 'content': data[-1]['system']},
            {'role': 'user', 'content': conversation[-1]['user']},
        ]
        for j in range(1, len(actions) + 1):
            s = '__'.join(str(x) for x in actions[: j])
            action_states.append(s)
        for a in action_states:
            rewards.append([Q[a][k] for k in Q[a]])
        for t in range(len(data)):
            dataset.append({'case': i, 'step': t, 'state': states[t], 'reward': rewards[t], 'action': actions_[t], 'final_state': final_state})
    dict2json(dataset, f'data/{args.system_set}/{args.task}_train.json')

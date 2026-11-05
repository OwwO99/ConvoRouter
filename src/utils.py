import json
import pickle
import pandas as pd


def read_json(file_path):
    file = open(file_path, 'r', encoding='utf-8')
    data = []
    for line in file.readlines():
        data.append(json.loads(line))
    return data


def save_json(data, save_path, add='w'):
    with open(save_path, add, encoding='utf-8') as f:
        for d in data:
            json_line = json.dumps(d, ensure_ascii=False)
            f.write(json_line + '\n')


def read_csv(file_path):
    df = pd.read_csv(file_path)
    data = []
    for i in range(df.shape[0]):
        data.append(dict(df.iloc[i]))
    return data


def save_csv(data, save_path):
    df = pd.DataFrame(data)
    with open(save_path, 'w', encoding='utf-8-sig', newline='') as f:
        df.to_csv(f, index=False, header=f.tell()==0)


def json2dict(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def dict2json(data, file_path):
    json_str = json.dumps(data, indent=4, ensure_ascii=False)
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json_file.write(json_str)


def read_pkl(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def save_pkl(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

"""
              ┏┓      ┏┓ + +
             ┏┛┻━━━━━━┛┻┓ + +
             ┃          ┃
             ┃    ━     ┃ ++ + + +
            ██████━██████ +
             ┃          ┃ +
             ┃    ┻     ┃
             ┃          ┃ + +
             ┗━┓      ┏━┛
               ┃      ┃
               ┃      ┃ + + + +
               ┃      ┃   
               ┃      ┃ + 　　　　神兽保佑,loss->0
               ┃      ┃        
               ┃      ┃  +
               ┃      ┗━━━━━┓ + +
               ┃            ┣┓
               ┃            ┏┛
               ┗━┓┓┏━━━━┳┓┏━┛ + + + +
                 ┃┫┫    ┃┫┫
                 ┗┻┛    ┗┻┛ + + + +

    author: abtion
    email: abtion@outlook.com
"""
import random

import pandas as pd
import os

LEN = 1024


def preproc():
    path = os.path.join('data', 'data.csv')
    data = pd.read_csv(path)
    data = data.dropna(axis=0, how='any')

    test_idxs = random.sample(list(range(len(data))), LEN)
    train_idxs = [i for i in range(len(data)) if i not in test_idxs]
    train_data = data.iloc[train_idxs, :].copy().reset_index(drop=True)
    test_data = data.iloc[test_idxs, :]

    # sample_indexes = list(train_data[train_data.iloc[:, 0] == 1].index)
    # while True:
    #     temp = float(train_data.iloc[:, 0].sum()) / train_data.shape[0]
    #     print(temp)
    #     if temp < 0.5:
    #
    #         _i = random.choice(sample_indexes)
    #         train_data.loc[train_data.shape[0]+1] = train_data.iloc[_i, :]
    #
    #     else:
    #         break
    print(len(train_data) + len(test_data) == len(data))
    train_data.to_csv(os.path.join('data', 'train.csv'), index=False)
    test_data.to_csv(os.path.join('data', 'test.csv'), index=False)


if __name__ == '__main__':
    preproc()

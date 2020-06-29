import codecs
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def yahoo_load():

    with codecs.open(f'../data/yahoo/train.txt', 'r', 'utf-8', errors='ignore') as f:
        train = pd.read_csv(f, delimiter=',', header=None)
        train.rename(columns={0: 'user', 1: 'item', 2: 'rate'}, inplace=True)
    for data in [train]:
        data.rate[data.rate < 4] = 0
        data.rate[data.rate >= 4] = 1
    train = train.values
    train, val = train_test_split(train, test_size=0.1, random_state=12345)
    train = train[train[:, 2] == 1, :]
    num_users, num_items = train[:, 0].max() + 1, train[:, 1].max() + 1
    _, item_fre = np.unique(train[:, 1], return_counts=True)
    _, user_act = np.unique(train[:, 0], return_counts=True)
    propensity = (item_fre/item_fre.max()) ** 0.5
    all_data = pd.DataFrame(np.zeros((num_users, num_items))).stack().reset_index()
    all_data = all_data.values[:, :2]
    train = train[:, :2]
    zero_data = np.array(list(set(map(tuple, all_data)) - set(map(tuple, train))), dtype=int)
    train = np.r_[np.c_[train, np.ones(train.shape[0])], np.c_[zero_data, np.zeros(zero_data.shape[0])]]
    propensity_score = np.zeros(train.shape[0])
    for i in range(train.shape[0]):
        propensity_score[i] = propensity[int(train[i, 1])]
    train = np.c_[train, propensity_score]
    with codecs.open(f'../data/yahoo/test.txt', 'r', 'utf-8', errors='ignore') as f:
        test = pd.read_csv(f, delimiter=',', header=None)
        test.rename(columns={0: 'user', 1: 'item', 2: 'rate'}, inplace=True)
    for data in [test]:
        data.rate[data.rate < 4] = 0
        data.rate[data.rate >= 4] = 1
    test = test.values
    test_rare = test[item_fre[test[:, 1]] < 250]
    test_active = test[user_act[test[:, 0]] > 10]
    np.save(file=f'../data/train.npy', arr=train)
    np.save(file=f'../data/test.npy', arr=test)
    np.save(file=f'../data/test_rare.npy', arr=test_rare)
    return train, test, test_rare, test_active, num_users, num_items

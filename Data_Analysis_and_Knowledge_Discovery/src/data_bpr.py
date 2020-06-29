import pandas as pd
from collections import defaultdict
import codecs
import numpy as np
from sklearn.model_selection import train_test_split

def yahoo_load_bpr():

    with codecs.open(f'../data/yahoo/train.txt', 'r', 'utf-8', errors='ignore') as f:
        train = pd.read_csv(f, delimiter=',', header=None)
        train.rename(columns={0: 'user', 1: 'item', 2: 'rate'}, inplace=True)
    for data in [train]:
        data.rate[data.rate < 4] = 0
        data.rate[data.rate >= 4] = 1
    train = train.values
    train, val = train_test_split(train, test_size=0.1, random_state=12345)
    train_ = train[train[:, 2] == 1, :]
    _, item_fre = np.unique(train[:, 1], return_counts=True)
    prop = (item_fre / item_fre.max()) ** 0.5
    _, user_act = np.unique(train[:, 0], return_counts=True)
    num_users, num_items = train[:, 0].max() + 1, train[:, 1].max() + 1
    with codecs.open(f'../data/yahoo/test.txt', 'r', 'utf-8', errors='ignore') as f:
        test = pd.read_csv(f, delimiter=',', header=None)
        test.rename(columns={0: 'user', 1: 'item', 2: 'rate'}, inplace=True)
    for data in [test]:
        data.rate[data.rate < 4] = 0
        data.rate[data.rate >= 4] = 1
    test = test.values
    test_rare = test[item_fre[test[:, 1]] < 500]
    test_active = test[user_act[test[:, 0]] > 10]
    train = defaultdict(list)
    num_ratings = train_.shape[0]
    for k in range(num_ratings):
        user, item = train_[k, 0], train_[k, 1]
        train[user].append([item])

    return num_ratings, num_users, num_items, train, test, prop

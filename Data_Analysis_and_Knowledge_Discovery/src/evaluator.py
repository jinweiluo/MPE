from typing import  List
import numpy as np
import pandas as pd
from metric import average_precision_at_k, dcg_at_k, recall_at_k


class AfterLearnModel:
    def __init__(self, name: str):
        self.user_embed = np.load(f'../logs/{name}/embeds/user_embed.npy')
        self.item_embed = np.load(f'../logs/{name}/embeds/item_embed.npy')

    def predict(self, users: np.array, items: np.array):
        user_emb = self.user_embed[users].reshape(1, self.user_embed.shape[1])
        item_emb = self.item_embed[items]
        scores = (user_emb @ item_emb.T).flatten()
        return scores


class Evaluator:
    def __init__(self, test: np.array, rare: bool, name: str):
        self.model = AfterLearnModel(name)
        self.users = test[:, 0]
        self.items = test[:, 1]
        self.ratings = 0.01 + 0.99 * test[:, 2]
        self.rare = rare
        self.name = name

    def evaluate(self, k: List[int] = [1, 3, 5]):
        results = {}
        metrics = {'DCG': dcg_at_k, 'Recall': recall_at_k, 'MAP': average_precision_at_k}

        for _k in k:
            for metric in metrics:
                results[f'{metric}@{_k}'] = []
        np.random.seed(12345)
        for user in set(self.users):
            indices = self.users == user
            items = self.items[indices]
            ratings = self.ratings[indices]
            scores = self.model.predict(users=np.int(user), items=items)
            for _k in k:
                for metric, metric_func in metrics.items():
                    results[f'{metric}@{_k}'].append(metric_func(ratings, scores, _k))
            self.results = pd.DataFrame(index=results.keys())
            self.results[f'{self.name}'] = list(map(np.mean, list(results.values())))
            if self.rare:
                self.results.to_csv(f'../logs/{self.name}/results/ranking_rare.csv')
            if not self.rare:
                self.results.to_csv(f'../logs/{self.name}/results/ranking.csv')


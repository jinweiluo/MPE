import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from model import PointwiseImplicitRecommender


class FmfTrainer:

    def __init__(
            self, latent_dim: int = 5, regu_lam: float = 1e-5, iters: int = 500,
            batch_size: int = 12, learning_rate: float = 0.1, num_users: int = 15400, num_items: int = 1000,
            weight: float = 1.0, mpe: bool = False):
        self.latent_dim = latent_dim
        self.regu_lam = regu_lam
        self.batch_size = batch_size
        self.iters = iters
        self.learning_rate = learning_rate
        self.num_users = num_users
        self.num_items = num_items
        self.weight = weight
        self.mpe = mpe
        os.makedirs(f'../logs/fmf/results', exist_ok=True)
        os.makedirs(f'../logs/fmf-mpe/results', exist_ok=True)

    def run(self, train, test, iters, batch_size):
        tf.set_random_seed(12345)
        ops.reset_default_graph()
        sess = tf.Session()
        rec = PointwiseImplicitRecommender(
            num_users=self.num_users, num_items=self.num_items,
            latent_dim=self.latent_dim, regu_lam=self.regu_lam, learning_rate=self.learning_rate, weight=self.weight)
        train_loss_list = []
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        np.random.seed(12345)
        train_one = train[train[:, 2] == 1]
        train_zero = train[train[:, 2] == 0]
        for i in np.arange(iters):
            idx_1 = np.random.choice(np.arange(int(train_one.shape[0]), dtype=int), size=int(batch_size/2))
            idx_0 = np.random.choice(np.arange(int(train_zero.shape[0]), dtype=int), size=int(batch_size/2))
            train_batch = np.r_[train_one[idx_1], train_zero[idx_0]]
            _, loss = sess.run([rec.apply_grads_fmf, rec.fmf_squre],
                               feed_dict={rec.users: train_batch[:, 0],
                                          rec.items: train_batch[:, 1],
                                          rec.labels: np.expand_dims(train_batch[:, 2], 1),
                                          rec.scores: np.expand_dims(train_batch[:, 3], 1)})
            train_loss_list.append(loss)
        if self.mpe:
            os.makedirs(f'../logs/fmf-mpe/embeds/', exist_ok=True)
            u_emb, i_emb = sess.run([rec.user_embeddings, rec.item_embeddings])
            np.save(file=f'../logs/fmf-mpe/embeds/user_embed.npy', arr=u_emb)
            np.save(file=f'../logs/fmf-mpe/embeds/item_embed.npy', arr=i_emb)
            os.makedirs(f'../logs/fmf-mpe/loss/', exist_ok=True)
            np.save(file=f'../logs/fmf-mpe/loss/train.npy', arr=np.array(train_loss_list))
            sess.close()
        else:
            os.makedirs(f'../logs/fmf/embeds/', exist_ok=True)
            u_emb, i_emb = sess.run([rec.user_embeddings, rec.item_embeddings])
            np.save(file=f'../logs/fmf/embeds/user_embed.npy', arr=u_emb)
            np.save(file=f'../logs/fmf/embeds/item_embed.npy', arr=i_emb)
            os.makedirs(f'../logs/fmf/loss/', exist_ok=True)
            np.save(file=f'../logs/fmf/loss/train.npy', arr=np.array(train_loss_list))
            sess.close()

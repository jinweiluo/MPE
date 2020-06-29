import os
import random
import numpy as np
import tensorflow as tf
from collections import defaultdict


class Bpr(object):

    def __init__(self, train, test, num_user, num_item, latent_dim,
                 Optimizer, learning_rate,
                 regu_lam, prop):
        self.train = train
        self.test = test
        self.num_user = num_user
        self.num_item = num_item
        self.latent_dim = latent_dim
        self.Optimizer = Optimizer
        self.learning_rate = learning_rate
        self.regu_lam = regu_lam
        self.prop = prop
        os.makedirs(f'../logs/bpr/results', exist_ok=True)
        os.makedirs(f'../logs/bpr/embeds/', exist_ok=True)
        self.items_of_user = defaultdict(set)
        self.num_rating = 0
        for u in range(0, self.num_user):
            for i in train[u]:
                self.items_of_user[u].add(i[0])
                self.num_rating += 1

        self.u = tf.placeholder(tf.int32, [None])
        self.i = tf.placeholder(tf.int32, [None])
        self.j = tf.placeholder(tf.int32, [None])
        self.labels2 = tf.placeholder(tf.float32, [None, 1], name='label_placeholder2')
        self.scores1 = tf.placeholder(tf.float32, [None, 1], name='score_placeholder')
        self.scores2 = tf.placeholder(tf.float32, [None, 1], name='score_placeholder')

        self.user_emb_w = tf.get_variable(
            'user_embeddings', shape=[self.num_user, self.latent_dim],
            initializer=tf.contrib.layers.xavier_initializer())
        self.item_emb_w = tf.get_variable(
            'item_embeddings', shape=[self.num_item, self.latent_dim],
            initializer=tf.contrib.layers.xavier_initializer())

        self.u_emb = tf.nn.embedding_lookup(self.user_emb_w, self.u)
        self.i_emb = tf.nn.embedding_lookup(self.item_emb_w, self.i)
        self.j_emb = tf.nn.embedding_lookup(self.item_emb_w, self.j)

        # calculate loss of the sample
        y_ui = tf.reduce_sum(tf.multiply(self.u_emb, self.i_emb), axis=1, keep_dims=True)
        y_uj = tf.reduce_sum(tf.multiply(self.u_emb, self.j_emb), axis=1, keep_dims=True)
        l2_reg = self.regu_lam * tf.add_n([tf.reduce_sum(tf.multiply(self.u_emb, self.u_emb)),
                                 tf.reduce_sum(tf.multiply(self.i_emb, self.i_emb)),
                                 tf.reduce_sum(tf.multiply(self.j_emb, self.j_emb))])
        bprloss = l2_reg - tf.reduce_mean(tf.log(tf.sigmoid(y_ui - y_uj)))
        self.apply_grads = self.Optimizer(self.learning_rate).minimize(bprloss)

    def build_model(self, test: np.array, iters=100, batch_size=32):
        self.batch_size = batch_size
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for iteration in range(iters):
                for _ in range(self.num_rating // 50 // self.batch_size):
                    uij_train = self.get_batch()
                    sess.run([self.apply_grads], feed_dict={
                             self.u: uij_train[:, 0],
                             self.i: uij_train[:, 1],
                             self.j: uij_train[:, 2]})
            u_emb, i_emb = sess.run([self.user_emb_w, self.item_emb_w])
            np.save(file=f'../logs/bpr/embeds/user_embed.npy', arr=u_emb)
            np.save(file=f'../logs/bpr/embeds/item_embed.npy', arr=i_emb)

    def get_batch(self):
        t = []
        for _ in range(self.batch_size):
            # sample a user
            _u = random.sample(range(0, self.num_user), 1)[0]
            while len(self.items_of_user[_u]) == 0:
                _u = random.sample(range(0, self.num_user), 1)[0]
            # sample a positive item
            _i = random.sample(self.items_of_user[_u], 1)[0]
            # sample a negative item
            _j = random.sample(range(0, self.num_item), 1)[0]
            while _j in self.items_of_user[_u]:
                _j = random.sample(range(1, self.num_item), 1)[0]
            t.append([_u, _i, _j])
        return np.asarray(t)

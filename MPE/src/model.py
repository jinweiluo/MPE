from __future__ import absolute_import, print_function
import numpy as np
import tensorflow as tf


class PointwiseImplicitRecommender:

    def __init__(self, num_users: np.array, num_items: np.array,
                 latent_dim: int, regu_lam: float, learning_rate: float, weight: float = 1.0) -> None:
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.regu_lam = regu_lam
        self.learning_rate = learning_rate
        self.weight = weight
        self.create_placeholders()
        self.build_graph()
        self.create_losses()
        self.add_optimizer()

    def create_placeholders(self):
        self.users = tf.placeholder(tf.int32, [None], name='user_placeholder')
        self.items = tf.placeholder(tf.int32, [None], name='item_placeholder')
        self.scores = tf.placeholder(tf.float32, [None, 1], name='score_placeholder')
        self.labels = tf.placeholder(tf.float32, [None, 1], name='label_placeholder')

    def build_graph(self):
        self.user_embeddings = tf.get_variable(
            'user_embeddings', shape=[self.num_users, self.latent_dim],
            initializer=tf.contrib.layers.xavier_initializer())
        self.item_embeddings = tf.get_variable(
            'item_embeddings', shape=[self.num_items, self.latent_dim],
            initializer=tf.contrib.layers.xavier_initializer())
        self.u_embed = tf.nn.embedding_lookup(self.user_embeddings, self.users)
        self.i_embed = tf.nn.embedding_lookup(self.item_embeddings, self.items)
        self.logits = tf.reduce_sum(tf.multiply(self.u_embed, self.i_embed), 1)
        self.preds = tf.sigmoid(tf.expand_dims(self.logits, 1), name='sigmoid_prediction')

    def create_losses(self):
        scores = tf.clip_by_value(
                self.scores, clip_value_min=0.01, clip_value_max=1.0)   # clip the propensity score not less than 0.01
        self.weighted_squre = tf.reduce_mean(
            (self.labels / scores) * tf.square(1. - self.preds) + (1 - self.labels / scores)
            * tf.square(self.preds))
        self.fmf_squre = tf.reduce_mean(
            self.weight * self.labels * tf.square(1. - self.preds) + (1 - self.labels) * self.scores
            * tf.square(self.preds))
        reg_term_embeds = tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(self.item_embeddings)
        self.loss = self.weighted_squre + self.regu_lam * reg_term_embeds
        self.loss_fmf = self.fmf_squre + self.regu_lam * reg_term_embeds

    def add_optimizer(self):
        self.apply_grads = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)
        self.apply_grads_fmf = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss_fmf)

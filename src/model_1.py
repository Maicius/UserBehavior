# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     model_1
   Author :        Xiaosong Zhou
   date：          2019/7/21
-------------------------------------------------
"""
__author__ = 'Xiaosong Zhou'


import tensorflow as tf
import datetime
import os
import numpy as np
from src.main import UserBehavior
from tensorflow import keras
from tensorflow.python.ops import summary_ops_v2


class RecommenderNetworkConfig(object):
    batch_size = 256
    MODEL_DIR = '../model'
    checkpoint_dir = os.path.join(MODEL_DIR, 'checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    train_dir = os.path.join(MODEL_DIR, 'summaries', 'train')
    test_dir = os.path.join(MODEL_DIR, 'summaries', 'eval')
    lr = 0.0001
    embed_dim = 40
    user_dim = 40
    item_dim = 44


class RecommenderNetwork(object):
    def __init__(self, config):
        self.config = config
        self.user_feature = tf.keras.layers.Input((40,), dtype='float32', name='user_feature')
        self.item_feature = tf.keras.layers.Input((44,), dtype='float32', name='item_feature')

        # 我这里开个全连接层，可训练
        self.user_fc = tf.keras.layers.Dense(self.config.embed_dim, name='user_fc', activation='relu')(self.user_feature)
        self.item_fc = tf.keras.layers.Dense(self.config.embed_dim, name='item_fc', activation='relu')(self.item_feature)

        inference = tf.keras.layers.Lambda(lambda layer: tf.reduce_sum(layer[0] * layer[1], axis=1), name="inference")(
            (self.user_fc, self.item_fc))
        inference = tf.keras.layers.Lambda(lambda layer: tf.expand_dims(layer, axis=1))(inference)
        self.model = tf.keras.Model(inputs=[self.user_feature, self.item_feature], outputs=[inference])
        self.model.summary()
        self.optimizer = tf.keras.optimizers.Adam(self.config.lr)
        # MSE损失，将计算值回归到评分
        self.ComputeLoss = tf.keras.losses.MeanSquaredError()
        self.ComputeMetrics = tf.keras.metrics.MeanAbsoluteError()
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        # Restore variables on creation if a checkpoint exists.
        self.checkpoint.restore(tf.train.latest_checkpoint(config.checkpoint_dir))


    def compute_loss(self, labels, logits):
        return tf.reduce_mean(tf.keras.losses.mse(labels, logits))

    def compute_metrics(self, labels, logits):
        return tf.keras.metrics.mae(labels, logits)

    @tf.function
    def train_step(self, x, y):
        # Record the operations used to compute the loss, so that the gradient
        # of the loss with respect to the variables can be computed.
        #         metrics = 0
        with tf.GradientTape() as tape:
            logits = self.model([x[0],
                                 x[1]], training=True)
            loss = self.ComputeLoss(y, logits)
            # loss = self.compute_loss(y, logits)
            self.ComputeMetrics(y, logits)
            # metrics = self.compute_metrics(y, logits)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, logits





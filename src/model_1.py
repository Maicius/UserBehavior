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
    model_dir = '../model'
    lr = 0.001


class RecommenderNetwork(object):
    def __init__(self, config):
        self.config = config
        self.user_feature = tf.keras.layers.Input((40,), dtype='int32', name='user_feature')
        self.item_feature = tf.keras.layers.Input((44,), dtype='int32', name='item_feature')


        inference = tf.keras.layers.Lambda(lambda layer: tf.reduce_sum(layer[0] * layer[1], axis=1), name="inference")(
            (self.user_embedding, self.item_feature_embedding))
        inference = tf.keras.layers.Lambda(lambda layer: tf.expand_dims(layer, axis=1))(inference)

        self.model = tf.keras.Model(inputs=[self.user_featureuser_feature, self.item_feature], outputs=[inference])
        self.model.summary()
        self.optimizer = tf.keras.optimizers.Adam(self.config.lr)
        # MSE损失，将计算值回归到评分
        self.ComputeLoss = tf.keras.losses.MeanSquaredError()
        self.ComputeMetrics = tf.keras.metrics.MeanAbsoluteError()
        if tf.io.gfile.exists(self.MODEL_DIR):
            #             print('Removing existing model dir: {}'.format(MODEL_DIR))
            #             tf.io.gfile.rmtree(MODEL_DIR)
            pass
        else:
            tf.io.gfile.makedirs(self.MODEL_DIR)

        train_dir = os.path.join(self.MODEL_DIR, 'summaries', 'train')
        test_dir = os.path.join(self.MODEL_DIR, 'summaries', 'eval')
        checkpoint_dir = os.path.join(self.MODEL_DIR, 'checkpoints')
        self.checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        # Restore variables on creation if a checkpoint exists.
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    def compute_loss(self, labels, logits):
        return tf.reduce_mean(tf.keras.losses.mse(labels, logits))

    def compute_metrics(self, labels, logits):
        return tf.keras.metrics.mae(labels, logits)

    def train_step(self, x, y):
        # Record the operations used to compute the loss, so that the gradient
        # of the loss with respect to the variables can be computed.
        #         metrics = 0
        with tf.GradientTape() as tape:
            logits = self.model([x[0],
                                 x[1]], training=True)
            loss = self.ComputeLoss(y, logits)
            # loss = self.compute_loss(labels, logits)
            self.ComputeMetrics(y, logits)
            # metrics = self.compute_metrics(labels, logits)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, logits





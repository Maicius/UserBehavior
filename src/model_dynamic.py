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
from tensorflow import keras
from tensorflow.python.ops import summary_ops_v2


class RecommenderNetworkConfig(object):
    train_batch_size = 512
    test_batch_size = 512
    MODEL_DIR = '../model'
    checkpoint_dir = os.path.join(MODEL_DIR, 'checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    train_dir = os.path.join(MODEL_DIR, 'summaries', 'train')
    test_dir = os.path.join(MODEL_DIR, 'summaries', 'eval')
    lr = 0.0001
    embed_dim = 32
    fc_dim = 32
    hidden_dim = 128
    user_dim = 40
    item_dim = 44
    gender_dim = 2
    age_dim = 10
    career_dim = 10
    income_dim = 12
    stage_dim = 11

    price_dim = 2231
    cate_1_dim = 111
    cate_dim = 8381
    brand_dim = 197784


class RecommenderNetwork(object):
    def __init__(self, config):
        self.config = config
        self.best_loss = 9999
        self.losses = {'train': [], 'test': []}
        user_gender, user_age, user_career, user_income, user_stage, \
        item_cate_1, item_cate, item_brand, item_price = self.get_inputs()

        gender_embed_layer, age_embed_layer, career_embed_layer, income_embed_layer, stage_embed_layer = self.\
            get_user_embedding(user_gender, user_age, user_career, user_income, user_stage)
        cate_1_embed_layer, cate_embed_layer, brand_embed_layer, price_embed_layer = self.\
            get_item_embedding(item_cate_1, item_cate, item_brand, item_price)

        _, user_combine_layer_flat = self.get_user_feature_layer(
            gender_embed_layer, age_embed_layer, career_embed_layer, income_embed_layer, stage_embed_layer
        )
        _, item_combine_layer_flat = self.get_item_feature_layer(
            cate_1_embed_layer, cate_embed_layer, brand_embed_layer, price_embed_layer
        )

        # inference
        inference = tf.keras.layers.Lambda(lambda layer: tf.reduce_sum(layer[0] * layer[1], axis=1), name="inference")(
            (user_combine_layer_flat, item_combine_layer_flat))
        inference = tf.keras.layers.Lambda(lambda layer: tf.expand_dims(layer, axis=1))(inference)
        self.model = tf.keras.Model(
            inputs=[user_gender, user_age, user_career, user_income, user_stage,
                    item_cate_1, item_cate, item_brand, item_price],
            outputs=[inference]
        )
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

    def get_inputs(self):
        user_gender = tf.keras.layers.Input(shape=(1,), dtype='int32', name='user_gender')
        user_age = tf.keras.layers.Input(shape=(1,), dtype='int32', name='user_age')
        user_career = tf.keras.layers.Input(shape=(1,), dtype='int32', name='user_career')
        user_income = tf.keras.layers.Input(shape=(1,), dtype='int32', name='user_income')
        user_stage = tf.keras.layers.Input(shape=(6,), dtype='int32', name='user_stage')

        item_cate_1 = tf.keras.layers.Input(shape=(1,), dtype='int32', name='item_cate_1')
        item_cate = tf.keras.layers.Input(shape=(1,), dtype='int32', name='item_cate')
        item_brand = tf.keras.layers.Input(shape=(1,), dtype='int32', name='item_brand')
        item_price = tf.keras.layers.Input(shape=(1,), dtype='int32', name='item_price')
        return user_gender, user_age, user_career, user_income, user_stage, item_cate_1, item_cate, item_brand, item_price

    def get_user_embedding(self, user_gender, user_age, user_career, user_income, user_stage):
        gender_embed_layer = tf.keras.layers.Embedding(self.config.gender_dim, self.config.embed_dim // 2,
                                                       input_length=1, name='gender_layer')(user_gender)
        age_embed_layer = tf.keras.layers.Embedding(self.config.age_dim, self.config.embed_dim // 2,
                                                    input_length=1, name='age_layer')(user_age)
        career_embed_layer = tf.keras.layers.Embedding(self.config.career_dim, self.config.embed_dim // 2,
                                                       input_length=1, name='career_layer')(user_career)
        income_embed_layer = tf.keras.layers.Embedding(self.config.income_dim, self.config.embed_dim // 2,
                                                       input_length=1, name='income_layer')(user_income)
        stage_embed_layer = tf.keras.layers.Embedding(self.config.stage_dim, self.config.embed_dim // 2,
                                                      input_length=6, name='stage_layer')(user_stage)
        stage_embed_layer = tf.keras.layers. \
            Lambda(lambda layer: tf.reduce_sum(layer, axis=1, keepdims=True))(stage_embed_layer)

        return gender_embed_layer, age_embed_layer, career_embed_layer, income_embed_layer, stage_embed_layer

    def get_item_embedding(self, item_cate_1, item_cate, item_brand, item_price):
        cate_1_embed_layer = tf.keras.layers.Embedding(self.config.cate_1_dim, self.config.embed_dim // 2,
                                                       input_length=1, name='cate_1_layer')(item_cate_1)
        cate_embed_layer = tf.keras.layers.Embedding(self.config.cate_dim, self.config.embed_dim * 2,
                                                     input_length=1, name='cate_layer')(item_cate)
        brand_embed_layer = tf.keras.layers.Embedding(self.config.brand_dim, self.config.embed_dim * 4,
                                                      input_length=1, name='brand_layer')(item_brand)
        price_embed_layer = tf.keras.layers.Embedding(self.config.price_dim, self.config.embed_dim // 2,
                                                      input_length=1, name='price_layer')(item_price)
        return cate_1_embed_layer, cate_embed_layer, brand_embed_layer, price_embed_layer

    def get_user_feature_layer(self, gender_embed_layer, age_embed_layer, career_embed_layer, income_embed_layer,
                               stage_embed_layer):
        gender_fc_layer = tf.keras.layers.Dense(self.config.fc_dim, name='gender_fc_layer',
                                                activation='relu')(gender_embed_layer)
        age_fc_layer = tf.keras.layers.Dense(self.config.fc_dim, name='age_fc_layer',
                                             activation='relu')(age_embed_layer)
        career_fc_layer = tf.keras.layers.Dense(self.config.fc_dim, name='career_fc_layer',
                                                activation='relu')(career_embed_layer)
        income_fc_layer = tf.keras.layers.Dense(self.config.fc_dim, name='income_fc_layer',
                                                activation='relu')(income_embed_layer)
        stage_fc_layer = tf.keras.layers.Dense(self.config.fc_dim, name='stage_fc_layer',
                                               activation='relu')(stage_embed_layer)
        # 第二层全连接
        user_combine_layer = tf.keras.layers.concatenate([gender_fc_layer, age_fc_layer, career_fc_layer,
                                                          income_fc_layer, stage_fc_layer], 2)
        user_combine_layer = tf.keras.layers.Dense(self.config.hidden_dim, activation='tanh')(user_combine_layer)

        user_combine_layer_flat = tf.keras.layers.Reshape([self.config.hidden_dim], name='user_combine_layer_flat')(
            user_combine_layer)
        return user_combine_layer, user_combine_layer_flat

    def get_item_feature_layer(self, cate_1_embed_layer, cate_embed_layer, brand_embed_layer, price_embed_layer):
        cate_1_fc_layer = tf.keras.layers.Dense(self.config.fc_dim, name='cate_1_fc_layer',
                                                activation='relu')(cate_1_embed_layer)
        cate_fc_layer = tf.keras.layers.Dense(self.config.fc_dim, name='cate_fc_layer',
                                              activation='relu')(cate_embed_layer)
        brand_fc_layer = tf.keras.layers.Dense(self.config.fc_dim, name='brand_fc_layer',
                                               activation='relu')(brand_embed_layer)
        price_fc_layer = tf.keras.layers.Dense(self.config.fc_dim, name='price_fc_layer',
                                               activation='relu')(price_embed_layer)
        # 第二层全连接
        item_combine_layer = tf.keras.layers.concatenate([cate_1_fc_layer, cate_fc_layer,
                                                          brand_fc_layer, price_fc_layer], 2)
        item_combine_layer = tf.keras.layers.Dense(self.config.hidden_dim, activation='tanh')(item_combine_layer)
        item_combine_layer_flat = tf.keras.layers.Reshape([self.config.hidden_dim], name='item_combine_layer_flat')(
            item_combine_layer)
        return item_combine_layer, item_combine_layer_flat

    @tf.function
    def train_step(self, x, y):
        # Record the operations used to compute the loss, so that the gradient
        # of the loss with respect to the variables can be computed.
        #         metrics = 0
        with tf.GradientTape() as tape:
            logits = self.model([x[0],
                                 x[1],
                                 x[2],
                                 x[3],
                                 x[4],
                                 x[5],
                                 x[6],
                                 x[7],
                                 x[8]], training=True)
            loss = self.ComputeLoss(y, logits)
            # loss = self.compute_loss(y, logits)
            self.ComputeMetrics(y, logits)
            # metrics = self.compute_metrics(y, logits)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, logits

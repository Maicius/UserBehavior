from src.main import UserBehavior
import tensorflow as tf
import datetime
import keras
import time
import os
import numpy as np
from sklearn.model_selection import train_test_split


class recommender_network(object):
    MODEL_DIR = '../model'

    def __init__(self, batch_size=256):
        self.batch_size = batch_size
        self.best_loss = 9999999
        self.losses = {'train': [], 'test': []}
        self.ub = UserBehavior()
        user_feature, item_feature = self.get_inputs()
        # 获得特征
        self.user_embedding, self.item_feature_embedding, self.user_item_score = self.ub.main()
        inference = keras.layers.Lambda(lambda layer: tf.reduce_sum(layer[0] * layer[1], axis=1), name="inference")(
            (self.user_embedding, self.item_feature_embedding))
        inference = keras.layers.Lambda(lambda layer: tf.expand_dims(layer, axis=1))(inference)

        self.model = keras.Model(inputs=[user_feature, item_feature], outputs=[inference])
        self.model.summary()
        self.optimizer = keras.optimizers.Adam(lr=0.001)
        # MSE损失，将计算值回归到评分
        self.ComputeLoss = keras.losses.MeanSquaredError()
        self.ComputeMetrics = keras.metrics.MeanAbsoluteError()
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
        return tf.reduce_mean(keras.losses.mse(labels, logits))

    def compute_metrics(self, labels, logits):
        return keras.metrics.mae(labels, logits)

    @tf.function
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

    @staticmethod
    def get_batches(Xs, ys, batch_size):
        for start in range(0, len(Xs), batch_size):
            end = min(start + batch_size, len(Xs))
            yield Xs[start:end], ys[start:end]

    def training(self, features, targets_values, epochs=5, log_freq=50):

        for epoch_i in range(epochs):
            # 将数据集分成训练集和测试集，随机种子不固定
            train_X, test_X, train_y, test_y = train_test_split(features,
                                                                targets_values,
                                                                test_size=0.2,
                                                                random_state=0)

            train_batches = self.get_batches(train_X, train_y, self.batch_size)
            batch_num = (len(train_X) // self.batch_size)

            train_start = time.time()
            #             with self.train_summary_writer.as_default():
            if True:
                start = time.time()
                # Metrics are stateful. They accumulate values and return a cumulative
                # result when you call .result(). Clear accumulated values with .reset_states()
                avg_loss = keras.metrics.Mean('loss', dtype=tf.float32)
                #                 avg_mae = tf.keras.metrics.Mean('mae', dtype=tf.float32)

                # Datasets can be iterated over like any other Python iterable.
                for batch_i in range(batch_num):
                    x, y = next(train_batches)
                    loss, logits = self.train_step([np.reshape(x.take(0, 1), [self.batch_size, 1]).astype('int32'),
                                                    np.reshape(x.take(1, 1), [self.batch_size, 1]).astype('int32'),
                                                    np.reshape(y, [self.batch_size, 1]).astype('int32')])

                    avg_loss(loss)
                    self.losses['train'].append(loss)
                    if tf.equal(self.optimizer.iterations % log_freq, 0):
                        rate = log_freq / (time.time() - start)
                        print('Step #{}\tEpoch {:>3} Batch {:>4}/{}   Loss: {:0.6f} mae: {:0.6f} ({} steps/sec)'.format(
                            self.optimizer.iterations.numpy(),
                            epoch_i,
                            batch_i,
                            batch_num,
                            loss, (self.ComputeMetrics.result()), rate))
                        # print('Step #{}\tLoss: {:0.6f} mae: {:0.6f} ({} steps/sec)'.format(
                        #     self.optimizer.iterations.numpy(), loss, (avg_mae.result()), rate))
                        avg_loss.reset_states()
                        self.ComputeMetrics.reset_states()
                        # avg_mae.reset_states()
                        start = time.time()

            train_end = time.time()
            print(
                '\nTrain time for epoch #{} ({} total steps): {}'.format(epoch_i + 1, self.optimizer.iterations.numpy(),
                                                                         train_end - train_start))
            #             with self.test_summary_writer.as_default():
            self.testing((test_X, test_y), self.optimizer.iterations)
            # self.checkpoint.save(self.checkpoint_prefix)
        self.export_path = os.path.join(self.MODEL_DIR, 'export')
        tf.saved_model.save(self.model, self.export_path)

    def get_inputs(self):
        user_feature = keras.layers.Input((40,), dtype='int32', name='user_feature')
        item_feature = keras.layers.Input((40,), dtype='int32', name='item_feature')

        return user_feature, item_feature



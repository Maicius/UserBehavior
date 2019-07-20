# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     run
   Author :        Xiaosong Zhou
   date：          2019/7/21
-------------------------------------------------
"""
__author__ = 'Xiaosong Zhou'
import tensorflow as tf
from src.model_1 import RecommenderNetworkConfig, RecommenderNetwork
from sklearn.model_selection import train_test_split
import time
import os


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
            avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
            #                 avg_mae = tf.keras.metrics.Mean('mae', dtype=tf.float32)

            # Datasets can be iterated over like any other Python iterable.
            for batch_i in range(batch_num):
                x, y = next(train_batches)
                loss, logits = network.train_step([np.reshape(x.take(0, 1), [self.batch_size, 1]).astype('int32'),
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



if __name__ == '__main__':
    config = RecommenderNetworkConfig()
    best_loss = 9999999
    losses = {'train': [], 'test': []}
    network = RecommenderNetwork(config)



    # print('hello')
    training()


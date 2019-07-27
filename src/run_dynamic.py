# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     run
   Author :        Xiaosong Zhou
   date：          2019/7/21
   动态载入数据，和之前的静态载入数据向量区别开来
-------------------------------------------------
"""
__author__ = 'Xiaosong Zhou'
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
print(sys.path)
import tensorflow as tf
import numpy as np
from src.model_dynamic import RecommenderNetworkConfig, RecommenderNetwork
from sklearn.model_selection import train_test_split
from src.main import UserBehavior
import time
import os
import pickle


def get_batches(Xs, ys, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]


def save_params(params):
    """
    Save parameters to file
    """
    pickle.dump(params, open('params.p', 'wb'))


def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open('params.p', mode='rb'))


def training(features, targets_values, epochs=5, log_freq=50):
    for epoch_i in range(epochs):
        # 将数据集分成训练集和测试集，随机种子不固定
        train_X, test_X, train_y, test_y = train_test_split(features,
                                                            targets_values,
                                                            test_size=0.2,
                                                            random_state=0)

        train_batches = get_batches(train_X, train_y, config.train_batch_size)
        batch_num = (len(train_X) // config.train_batch_size)

        train_start = time.time()
        #             with self.train_summary_writer.as_default():
        if True:
            start = time.time()
            # Metrics are stateful. They accumulate values and return a cumulative
            # result when you call .result(). Clear accumulated values with .reset_states()
            avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
            # avg_mae = tf.keras.metrics.Mean('mae', dtype=tf.float32)

            # Datasets can be iterated over like any other Python iterable.
            for batch_i in range(batch_num):

                x, y = next(train_batches)
                stages = np.zeros([config.train_batch_size, 6])
                for i in range(config.train_batch_size):
                    content = x.take(4,1)[i]
                    content_nums = list(map(int, content[1:-1].split(',')))
                    stages[i] = content_nums
                if batch_i == 1:
                    print('hi')
                loss, logits = network.train_step([np.reshape(x.take(0, 1), [config.train_batch_size, 1]).astype(np.float32),
                                                   np.reshape(x.take(1, 1), [config.train_batch_size, 1]).astype(np.float32),
                                                   np.reshape(x.take(2, 1), [config.train_batch_size, 1]).astype(np.float32),
                                                   np.reshape(x.take(3, 1), [config.train_batch_size, 1]).astype(np.float32),
                                                   stages.astype(np.float32),
                                                   np.reshape(x.take(5, 1), [config.train_batch_size, 1]).astype(np.float32),
                                                   np.reshape(x.take(6, 1), [config.train_batch_size, 1]).astype(np.float32),
                                                   np.reshape(x.take(7, 1), [config.train_batch_size, 1]).astype(np.float32),
                                                   np.reshape(x.take(8, 1), [config.train_batch_size, 1]).astype(np.float32)],
                                                  np.reshape(y, [config.train_batch_size, 1]).astype(np.float32))
                avg_loss(loss)
                network.losses['train'].append(loss)
                if tf.equal(network.optimizer.iterations % log_freq, 0):
                    rate = log_freq / (time.time() - start)
                    print('Step #{}\tEpoch {:>3} Batch {:>4}/{}   Loss: {:0.6f} mae: {:0.6f} ({} steps/sec)'.format(
                        network.optimizer.iterations.numpy(),
                        epoch_i,
                        batch_i,
                        batch_num,
                        loss, (network.ComputeMetrics.result()), rate))
                    # print('Step #{}\tLoss: {:0.6f} mae: {:0.6f} ({} steps/sec)'.format(
                    #     network.optimizer.iterations.numpy(), loss, (avg_mae.result()), rate))
                    avg_loss.reset_states()
                    network.ComputeMetrics.reset_states()
                    # avg_mae.reset_states()
                    start = time.time()

        train_end = time.time()
        print(
            '\nTrain time for epoch #{} ({} total steps): {}'.format(epoch_i + 1, network.optimizer.iterations.numpy(),
                                                                     train_end - train_start))
        #             with self.test_summary_writer.as_default():
        testing((test_X, test_y), network.optimizer.iterations)

        # self.checkpoint.save(self.checkpoint_prefix)
    export_path = os.path.join(config.MODEL_DIR, 'export')
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    tf.saved_model.save(network.model, export_path)


def testing(test_dataset, step_num):
    test_X, test_y = test_dataset
    test_batches = get_batches(test_X, test_y, config.test_batch_size)
    """Perform an evaluation of `model` on the examples from `dataset`."""
    avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    # avg_mae = tf.keras.metrics.Mean('mae', dtype=tf.float32)

    batch_num = (len(test_X) // config.test_batch_size)
    print("test_batch_num: " + str(batch_num))
    for batch_i in range(batch_num):
        print("test batch: "+ str(batch_i))
        x, y = next(test_batches)
        stages = np.zeros([config.test_batch_size, 6])
        for i in range(config.test_batch_size):
            content = x.take(4, 1)[i]
            content_nums = list(map(int, content[1:-1].split(',')))
            stages[i] = content_nums
        logits = network.model([np.reshape(x.take(0, 1), [config.test_batch_size, 1]).astype(np.float32),
                                np.reshape(x.take(1, 1), [config.test_batch_size, 1]).astype(np.float32),
                                np.reshape(x.take(2, 1), [config.test_batch_size, 1]).astype(np.float32),
                                np.reshape(x.take(3, 1), [config.test_batch_size, 1]).astype(np.float32),
                                stages.astype(np.float32),
                                np.reshape(x.take(5, 1), [config.test_batch_size, 1]).astype(np.float32),
                                np.reshape(x.take(6, 1), [config.test_batch_size, 1]).astype(np.float32),
                                np.reshape(x.take(7, 1), [config.test_batch_size, 1]).astype(np.float32),
                                np.reshape(x.take(8, 1), [config.test_batch_size, 1]).astype(np.float32)],
                               training=False)
        test_loss = network.ComputeLoss(y.astype('float32'), logits)
        avg_loss(test_loss)
        # 保存测试损失
        network.losses['test'].append(test_loss)
        network.ComputeMetrics(y.astype('float32'), logits)

    print('Model test set loss: {:0.6f} mae: {:0.6f}'.format(avg_loss.result(), network.ComputeMetrics.result()))
    # print('Model test set loss: {:0.6f} mae: {:0.6f}'.format(avg_loss.result(), avg_mae.result()))
    #         summary_ops_v2.scalar('loss', avg_loss.result(), step=step_num)
    #         summary_ops_v2.scalar('mae', self.ComputeMetrics.result(), step=step_num)
    # summary_ops_v2.scalar('mae', avg_mae.result(), step=step_num)
    global best_loss
    if avg_loss.result() < best_loss:
        best_loss = avg_loss.result()
        print("best loss = {}".format(best_loss))
        network.checkpoint.save(config.checkpoint_prefix)


def forward(self, xs):
    predictions = self.model(xs)
    # logits = tf.nn.softmax(predictions)

    return predictions

if __name__ == '__main__':
    config = RecommenderNetworkConfig()
    ub = UserBehavior()
    best_loss = 9999999
    losses = {'train': [], 'test': []}
    network = RecommenderNetwork(config)
    # 获得特征
    train_df, target_df = ub.chunk_load_train_data()
    train_data = train_df.values

    if tf.io.gfile.exists(config.MODEL_DIR):
        #             print('Removing existing model dir: {}'.format(MODEL_DIR))
        #             tf.io.gfile.rmtree(MODEL_DIR)
        pass
    else:
        tf.io.gfile.makedirs(config.MODEL_DIR)
    training(train_df.values, target_df.values, epochs=20)

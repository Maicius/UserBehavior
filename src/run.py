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
import numpy as np
from src.model_1 import RecommenderNetworkConfig, RecommenderNetwork
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

        train_batches = get_batches(train_X, train_y, config.batch_size)
        batch_num = (len(train_X) // config.batch_size)

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
                x_user, x_item = zip(*x)
                x_user = np.mat(list(x_user))
                x_item = np.mat(list(x_item))
                input_x_user = np.reshape(np.array(x_user), [config.batch_size, config.user_dim])
                input_x_item = np.reshape(np.array(x_item), [config.batch_size, config.item_dim])
                input_y = np.reshape(np.array(y), [config.batch_size, 1])
                input_x = [input_x_user.astype('float32'), input_x_item.astype('float32')]
                loss, logits = network.train_step(input_x, input_y.astype('float32'))

                avg_loss(loss)
                losses['train'].append(loss)
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
    test_batches = get_batches(test_X, test_y, config.batch_size)
    """Perform an evaluation of `model` on the examples from `dataset`."""
    avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    # avg_mae = tf.keras.metrics.Mean('mae', dtype=tf.float32)

    batch_num = (len(test_X) // config.batch_size)
    for batch_i in range(batch_num):
        x, y = next(test_batches)
        x_user, x_item = zip(*x)
        x_user = np.mat(list(x_user))
        x_item = np.mat(list(x_item))
        input_x_user = np.reshape(np.array(x_user), [config.batch_size, config.user_dim])
        input_x_item = np.reshape(np.array(x_item), [config.batch_size, config.item_dim])
        input_y = np.reshape(np.array(y), [config.batch_size, 1])
        input_x = [input_x_user.astype('float32'), input_x_item.astype('float32')]
        logits = network.model(input_x, input_y.astype('float32'))
        test_loss = network.ComputeLoss(input_y.astype('float32'), logits)
        avg_loss(test_loss)
        # 保存测试损失
        losses['test'].append(test_loss)
        network.ComputeMetrics(input_y.astype('float32'), logits)

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
    user_embedding, item_feature_embedding, user_item_score = ub.load_train_vector()
    # model_input_x = []
    # model_input_x.append(user_embedding)
    # model_input_x.append(item_feature_embedding)
    model_input_x = list(zip(user_embedding, item_feature_embedding))

    if tf.io.gfile.exists(config.MODEL_DIR):
        #             print('Removing existing model dir: {}'.format(MODEL_DIR))
        #             tf.io.gfile.rmtree(MODEL_DIR)
        pass
    else:
        tf.io.gfile.makedirs(config.MODEL_DIR)

    # checkpoint = tf.train.Checkpoint(model=network.model, optimizer=network.optimizer)
    # Restore variables on creation if a checkpoint exists.
    # network.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    training(model_input_x, user_item_score, epochs=5)

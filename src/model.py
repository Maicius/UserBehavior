from src.main import UserBehavior
import tensorflow as tf
import datetime
import keras
import time
import os

class recommender_network(object):
    MODEL_DIR = '../model'
    def __init__(self, batch_size=256):
        self.batch_size = batch_size
        self.best_loss = 9999
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

    def get_inputs(self):
        user_feature = keras.layers.Input((40,), dtype='int32', name='user_feature')
        item_feature = keras.layers.Input((40,), dtype='int32', name='item_feature')

        return user_feature, item_feature

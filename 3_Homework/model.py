"""Module defines the model"""
import tensorflow as tf

class AtariNet:
    """ This creates a Deep Q-Network for the Atari Learning Environment """
    def __init__(self, input_val, input_shape, action_num, learning_rate, name='AtariNet'):
        self.input_val = input_val
        self.input_shape = input_shape
        self.action_num = action_num
        self.actions = tf.placeholder(tf.float32, [None, self.action_num], name='actions')
        self.target_q = tf.placeholder(tf.float32, [None], name='target_q')
        # create the conv-net
        with tf.variable_scope(name):
            # conv layer 1
            self.conv_1 = tf.keras.layers.Conv2D(32, 8, 4,
                                                 input_shape=input_shape, activation=tf.nn.relu)(self.input_val)
            # conv layer 2
            self.conv_2 = tf.keras.layers.Conv2D(64, 4, 2, activation=tf.nn.relu)(self.conv_1)
            # conv layer 3
            self.conv_3 = tf.keras.layers.Conv2D(64, 3, 1, activation=tf.nn.relu)(self.conv_2)
            # flatten
            self.flat = tf.keras.layers.Flatten()(self.conv_3)
            # dense layer
            self.dense_1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(self.flat)
            # output layer
            self.output = tf.keras.layers.Dense(action_num)(self.dense_1)
        tf.identity(self.output, name='output')

        # saver
        self.saver = tf.train.Saver()
        # q-values
        self.q_val = tf.reduce_sum(tf.multiply(self.output, self.actions), axis=1)
        # huber loss
        self.loss = tf.losses.huber_loss(self.q_val, self.target_q)
        # optimizer
        self.org_optimizer = tf.train.AdamOptimizer(learning_rate)
        # gradient clipping
        self.optimizer = tf.contrib.estimator.clip_gradients_by_norm(self.org_optimizer, clip_norm=1.0)
        # minimize loss using training op
        self.train_op = self.optimizer.minimize(self.loss)

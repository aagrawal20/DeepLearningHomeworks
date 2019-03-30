import tensorflow as tf
import numpy as np

class AtariNet:
    def __init__(self, x, input_shape, action_num, learning_rate, name='AtariNet'):
        self.x = x
        self.input_shape = input_shape
        self.action_num = action_num
        self.actions = tf.placeholder(tf.float32, [None, self.action_num], name='actions')
        self.target_Q = tf.placeholder(tf.float32, [None], name='target_q')
        #self.state_actions = tf.placeholder(tf.float32, [None], name='state_actions')
        
        with tf.variable_scope(name):
            

            self.conv_1 = tf.keras.layers.Conv2D(32, 8, 4, input_shape=input_shape, activation=tf.nn.relu)(self.x)
            self.conv_2 = tf.keras.layers.Conv2D(64, 4, 2, activation=tf.nn.relu)(self.conv_1)
            self.conv_3 = tf.keras.layers.Conv2D(64, 3, 1, activation=tf.nn.relu)(self.conv_2)
            self.flat = tf.keras.layers.Flatten()(self.conv_3)
            self.dense_1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(self.flat)
            self.output = tf.keras.layers.Dense(action_num)(self.dense_1)
        tf.identity(self.output, name='output')


        self.saver = tf.train.Saver()
        self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions), axis=1)
        self.loss = tf.losses.huber_loss(self.Q, self.target_Q)
            # tf.reduce_mean(tf.square(self.target_Q - self.Q)) 
        self.org_optimizer = tf.train.AdamOptimizer(learning_rate)
        self.optimizer = tf.contrib.estimator.clip_gradients_by_norm(self.org_optimizer, clip_norm=1.0)
        self.train_op = self.optimizer.minimize(self.loss)

        
        
    def loss_optimize(self, session, cur_state_batch, actions, target_qvals):
        loss, _, Q = session.run([self.loss, self.train_op, self.Q], feed_dict={input:cur_state_batch, self.target_Q: target_qvals, self.actions:actions[0]})
        return loss, Q
        

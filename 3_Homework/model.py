import tensorflow as tf
import numpy as np

class AtariNet:
    def __init__(self, x, input_shape, action_num, learning_rate, name='AtariNet'):
        self.x = x
        self.input_shape = input_shape
        self.action_num = action_num
        
        with tf.variable_scope(name):
            self.actions = tf.placeholder(tf.float32, [None, self.action_num], name='actions')
            self.target_Q = tf.placeholder(tf.float32, [None], name='target')

            self.conv_1 = tf.keras.layers.Conv2D(32, 8, 4, input_shape=input_shape, activation=tf.nn.relu)(self.x)
            self.conv_2 = tf.keras.layers.Conv2D(64, 4, 2, activation=tf.nn.relu)(self.conv_1)
            self.conv_3 = tf.keras.layers.Conv2D(64, 3, 1, activation=tf.nn.relu)(self.conv_2)
            self.flat = tf.keras.layers.Flatten()(self.conv_3)
            self.dense_1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(self.flat)
            self.output = tf.keras.layers.Dense(action_num)(self.dense_1)
            

            # self.global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
            self.saver = tf.train.Saver()
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions), axis=1)
            self.loss = tf.losses.huber_loss(self.target_Q, self.Q)
            # tf.reduce_mean(tf.square(self.target_Q - self.Q)) 
            self.train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)
        
        tf.identity(self.output, name='output')

# def atari_net(x, input_shape, action_num):
    
#     conv_1 = tf.keras.layers.Conv2D(32, 8, 4, input_shape=input_shape, activation=tf.nn.relu)(x)
#     conv_2 = tf.keras.layers.Conv2D(64, 4, 2, activation=tf.nn.relu)(conv_1)
#     conv_3 = tf.keras.layers.Conv2D(64, 3, 1, activation=tf.nn.relu)(conv_2)
#     flat = tf.keras.layers.Flatten()(conv_3)
#     dense_1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(flat)
#     output = tf.keras.layers.Dense(action_num)(dense_1)
#     tf.identity(output, name='output')
    
#     return output


# def dqn_gradients(expected_state_action_values, state_action_values, learning_rate, gamma=0.99, grad_norm_clipping=1.0):
#     """
#     computes gradients for dqn
#     :param replay_memory: replay memory
#     :param transition: transition values
#     :param policy_model: (callable): mapping of `obs` to q-values
#     :param target_model: target model
#     :param batch_size: batch size
#     :param gamma: gamma hyper parameter
#     :param grad_norm_clipping: gradient norm clipping hyper parameter
#     :return: loss and gradients
#     """

#     # training and saving functionality
#     global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    
#     # compute Huber loss on TD-error
#     # td_error = state_action_values - expected_state_action_values
    
#     loss = tf.losses.huber_loss(expected_state_action_values, state_action_values)
#     optimizer = tf.train.RMSPropOptimizer(learning_rate)
#     train_op = optimizer.minimize(loss, global_step=global_step_tensor)
    
#     # gradients = tape.gradient(loss, policy_model.trainable_variables)
#     # saver
    
#     saver = tf.train.Saver()
    
#     # # clip gradients
#     # for i, grad in enumerate(gradients):
#     #     if grad is not None:
#     #         gradients[i] = tf.clip_by_norm(grad, grad_norm_clipping)

#     return loss, train_op, global_step_tensor, saver

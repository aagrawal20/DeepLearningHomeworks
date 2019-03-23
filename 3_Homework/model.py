import tensorflow as tf
import numpy as np

def atari_net(x, input_shape, action_num):
    
    conv_1 = tf.keras.layers.Conv2D(32, 8, 4, input_shape=input_shape, activation=tf.nn.relu)(x)
    conv_2 = tf.keras.layers.Conv2D(64, 4, 2, activation=tf.nn.relu)(conv_1)
    conv_3 = tf.keras.layers.Conv2D(64, 3, 1, activation=tf.nn.relu)(conv_2)
    flat = tf.keras.layers.Flatten()(conv_3)
    dense_1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(flat)
    output = tf.keras.layers.Dense(action_num)(dense_1)
    tf.identity(output, name='output')
    
    return output


def dqn_gradients(replay_memory, transition, policy_model, target_model, batch_size, learning_rate, gamma=0.99, grad_norm_clipping=1.0):
    """
    computes gradients for dqn
    :param replay_memory: replay memory
    :param transition: transition values
    :param policy_model: (callable): mapping of `obs` to q-values
    :param target_model: target model
    :param batch_size: batch size
    :param gamma: gamma hyper parameter
    :param grad_norm_clipping: gradient norm clipping hyper parameter
    :return: loss and gradients
    """
    # before enough transitions are collected to form a batch
    if len(replay_memory) < batch_size:
        return None, None

    # prepare training batch
    transitions = replay_memory.sample(batch_size)
    batch = transition(*zip(*transitions))
    next_states = np.array(batch.next_state, dtype=np.float32)
    state_batch = np.array(batch.state, dtype=np.float32)
    action_batch = np.array(batch.action, dtype=np.int64)
    reward_batch = np.array(batch.reward)

    # with tf.GradientTape() as tape:
    # calculate value from taking action
    action_idxs = np.stack([np.arange(batch_size, dtype=np.int32), action_batch], axis=1)
    state_action_values = tf.gather_nd(policy_model(state_batch), action_idxs)
    
    # calculate best value at next state
    next_state_values = tf.reduce_max(target_model(next_states), axis=1)

    # compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # training and saving functionality
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    
    # compute Huber loss on TD-error
    # td_error = state_action_values - expected_state_action_values
    
    loss = tf.losses.huber_loss(expected_state_action_values, state_action_values)
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    train_op = train_op = optimizer.minimize(loss, global_step=global_step_tensor)
    
    # gradients = tape.gradient(loss, policy_model.trainable_variables)
    # saver
    
    saver = tf.train.Saver()
    
    # # clip gradients
    # for i, grad in enumerate(gradients):
    #     if grad is not None:
    #         gradients[i] = tf.clip_by_norm(grad, grad_norm_clipping)

    return loss, train_op, global_step_tensor, saver

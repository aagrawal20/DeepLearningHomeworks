import argparse
import atari_wrappers
import numpy as np
import random
import tensorflow as tf
from util import select_eps_greedy_action, ReplayMemory, Transition
from model import atari_net

# setup parser
parser = argparse.ArgumentParser(description='Train Atari Agent.')
parser.add_argument(
    '--model_dir',
    type=str,
    default='/work/cse496dl/atendle/hw_3_logs/',
    help='directory where model graph and weights are saved')
parser.add_argument('--batch_size', type=int, default=64, help='mini batch size for training')
parser.add_argument('--ep_num', type=int, default=100, help='number of episodes to run')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer')
parser.add_argument('--target_update', type=float, default=10, help='Frequency of Target update step')

# setup parser arguments
args = parser.parse_args()

# load environment
env = atari_wrappers.wrap_deepmind(atari_wrappers.make_atari('SeaquestNoFrameskip-v4'), frame_stack=True)
# get shape of actions
NUM_ACTIONS = env.action_space.n
# get shape of observations
OBS_SHAPE = env.observation_space.shape

# setup placeholders
input = tf.placeholder(tf.float32, [None, 84, 84, 4], name='input_placeholder')

# setup hyperparameters
TARGET_UPDATE_STEP_FREQ = args.target_update
BATCH_SIZE = args.batch_size
EPISODE_NUM = args.ep_num
REPLAY_BUFFER_SIZE = 1000000
LEARNING_RATE = args.learning_rate

# setup policy and target model, replay memory, and optimizer
policy_model = atari_net(input, OBS_SHAPE, NUM_ACTIONS)
target_model = atari_net(input, OBS_SHAPE, NUM_ACTIONS)
replay_memory = ReplayMemory(REPLAY_BUFFER_SIZE)
loss, train_op, global_step_tensor, saver = dqn_gradients(replay_memory, Transition, policy_model, target_model, BATCH_SIZE)


with tf.Session() as session:
    # initialize variables
    session.run(tf.global_variables_initializer())
    step = 0
    score_list = [] 
    for episode in range(EPISODE_NUM):

        # initialize environment
        prev_observation = env.reset()
        observation, reward, done, _ = env.step(random.randrange(NUM_ACTIONS))
        print("Episode {}: Reward {}".format(episode,reward))
        done = False
        ep_score = 0.

        while not done: # until the episode ends

            # select and perform an action
            prepped_obs = np.expand_dims(np.array(observation, dtype=np.float32), axis=0)
            action = select_eps_greedy_action(policy_model, prepped_obs, step, NUM_ACTIONS)
            observation, reward, done, _ = env.step(action)

            # add to memory
            replay_memory.push(prev_observation, action, observation, reward)
            prev_observation = observation

            #TODO: train model
            # _ = session.run([train_op], feed)
            # if grads is not None:
                # optimizer.apply_gradients(zip(grads, policy_model.trainable_variables))
            # increment counters
            ep_score += reward
            step += 1

        #TODO: update the target network, copying all variables in DQN
        # if episode % TARGET_UPDATE_STEP_FREQ == 0:
        #     target_model.set_weights(policy_model.get_weights())

        print("Episode {} achieved score {} at {} training steps".format(episode, ep_score, step))
        score_list.append(ep_score)

    avg_score = (sum(score_list))/(len(score_list))
    print("Average episode score: {}".format(avg_score))
    print("Top score for all episodes: {}".format(max(score_list)))
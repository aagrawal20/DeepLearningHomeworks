import argparse
import atari_wrappers
import numpy as np
import random
import tensorflow as tf
from util import select_eps_greedy_action, ReplayMemory, Transition
from model import AtariNet

# setup parser
parser = argparse.ArgumentParser(description='Train Atari Agent.')
parser.add_argument(
    '--model_dir',
    type=str,
    default='/work/cse496dl/teams/Dropouts/3_Homework/agent_1/',
    help='directory where model graph and weights are saved')
parser.add_argument('--batch_size', type=int, default=64, help='mini batch size for training')
parser.add_argument('--ep_num', type=int, default=100, help='number of episodes to run')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for optimizer')
parser.add_argument('--target_update', type=float, default=500, help='Frequency of Target update step')
parser.add_argument('--eps_start', type=float, default = 1.)
parser.add_argument('--eps_end', type=float, default = 0.01)
parser.add_argument('--eps_decay', type=int, default=100000, help='Decay for the epsilon threshold')
parser.add_argument('--max_steps', type=int, default=50000, help='max steps per episode')

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
expected_vals = tf.placeholder(tf.float32, [None, 84, 84, 4] )
state_vals = tf.placeholder(tf.float32, [None, 84, 84, 4])

# setup hyperparameters
TARGET_UPDATE_STEP_FREQ = args.target_update
BATCH_SIZE = args.batch_size
EPS_START = args.eps_start
EPS_END = args.eps_end
EPS_DECAY = args.eps_decay
EPISODE_NUM = args.ep_num
REPLAY_BUFFER_SIZE = 50000
max_steps = args.max_steps
LEARNING_RATE = args.learning_rate
GAMMA = 0.99

# setup policy and target model, replay memory, and optimizer
policy_model = AtariNet(input, OBS_SHAPE, NUM_ACTIONS, LEARNING_RATE)
target_model = AtariNet(input, OBS_SHAPE, NUM_ACTIONS, LEARNING_RATE)
replay_memory = ReplayMemory(REPLAY_BUFFER_SIZE)
# loss, train_op, global_step_tensor, saver = dqn_gradients(expected_vals, state_vals, BATCH_SIZE, LEARNING_RATE)

print("Batch Size: {}".format(BATCH_SIZE))
print("Episodes: {}".format(EPISODE_NUM))
print("Target Update Freq: {}".format(TARGET_UPDATE_STEP_FREQ))
print("========================\n")
print("Epsilon Start: {}".format(EPS_START))
print("Epsilon End: {}".format(EPS_END))
print("Epsilon Decay: {}".format(EPS_DECAY))
print("========================\n")

with tf.Session() as session:
    # initialize variables
    session.run(tf.global_variables_initializer())
    step = 0
    score_list = [] 
    exploit= 0 
    explore = 0
    
    while len(replay_memory) < BATCH_SIZE:
        prev_observation = env.reset()
        observation, reward, done, _ = env.step(random.randrange(NUM_ACTIONS))
        
        prepped_obs = np.expand_dims(np.array(observation, dtype=np.float32), axis=0)
        action, count_explore, count_exploit = select_eps_greedy_action(session, input, policy_model, prepped_obs, step, NUM_ACTIONS, EPS_START, EPS_END, EPS_DECAY, exploit, explore)
        observation, reward, done, info = env.step(action)
        # add to memory
        print("Filling Replay Memory.", end='\r')
        replay_memory.push(prev_observation, action, observation, reward)
    
    print("\n====================\n")
    print("Training Start\n")
    for episode in range(EPISODE_NUM):
        print("------------------")
        print("| Episode: {}".format(episode))
        # initialize environment
        prev_observation = env.reset()
        observation, reward, done, _ = env.step(random.randrange(NUM_ACTIONS))
        done = False
        ep_score = 0
        steps = 0
        exploit= 0 
        explore = 0
        while steps < 100: # until the episode ends
            steps += 1
            
            
            # select and perform an action
            prepped_obs = np.expand_dims(np.array(observation, dtype=np.float32), axis=0)
            action, count_explore, count_exploit = select_eps_greedy_action(session, input, policy_model, prepped_obs, step, NUM_ACTIONS, EPS_START, EPS_END, EPS_DECAY, exploit, explore)
            observation, reward, done, info = env.step(action)
            # add to memory
            replay_memory.push(prev_observation, action, observation, reward)
            prev_observation = observation

            # before enough transitions are collected to form a batch
            if len(replay_memory) < BATCH_SIZE:
                break

            # prepare training batch
            transitions = replay_memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))
            next_states = np.array(batch.next_state, dtype=np.float32)
            state_batch = np.array(batch.state, dtype=np.float32)
            action_batch = np.array(batch.action, dtype=np.int64)
            reward_batch = np.array(batch.reward)

            # state values
            state_output = session.run([policy_model.output], feed_dict={input:state_batch})

            # calculate best value at next state
            next_state_output = session.run([policy_model.output], feed_dict={input:next_states})
            next_state_values = np.amax(next_state_output[0], axis=1)
            # compute the expected Q values
            expected_state_action_values = (next_state_values * GAMMA) + reward_batch
            # optimize
            loss, _ = session.run([policy_model.loss, policy_model.train_op], feed_dict={input: state_batch,
                                                                                        policy_model.target_Q: expected_state_action_values,
                                                                                        policy_model.actions: state_output[0]})

            ep_score += reward
            step += 1
            exploit= count_exploit 
            explore = count_explore

            
        #update the target network, copying all variables in DQN
        if episode % TARGET_UPDATE_STEP_FREQ == 0:
            # get trainable variables
            trainable_vars = tf.trainable_variables()
            # length of trainable variables
            total_vars = len(trainable_vars)
            # list to hold all operators
            ops = []

            # iterate through policy model weights
            policy_model_weights = trainable_vars[0:total_vars//2]
            for idx, var in enumerate(policy_model_weights):
                # get target model weights
                target_model_weights = trainable_vars[idx + total_vars//2]
                # assign policy model weights to target model weights
                ops.append(target_model_weights.assign((var.value())))

            # run session to transfer weights
            for op in ops:
                session.run(op)
                
        print("| Steps: {}".format(steps))
        print("| Score: {}".format(ep_score))
        print("| Expore: {}".format(count_explore))
        print("| Exploit: {}".format(count_exploit))
        print("------------------")
        # print("Episode {} achieved score {} at {} training steps\n".format(episode, ep_score, step))
        score_list.append(ep_score)

    avg_score = (sum(score_list))/(len(score_list))
    print("\nAverage episode score: {}".format(avg_score))
    print("Top score for all episodes: {}".format(max(score_list)))
    print("Total steps taken: {}".format(step))
    policy_model.saver.save(session, args.model_dir + "policy/" + "homework_3")
    target_model.saver.save(session, args.model_dir + "target/" + "homework_3")

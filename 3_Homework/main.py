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
    default='/work/cse496dl/teams/Dropouts/3_Homework/test_agent/',
    help='directory where model graph and weights are saved')
parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for optimizer')
parser.add_argument('--target_update', type=int, default=10, help='Frequency of Target update step')
parser.add_argument('--eps_start', type=float, default = 1.)
parser.add_argument('--eps_end', type=float, default = 0.01)
parser.add_argument('--eps_decay', type=int, default=100000, help='Decay for the epsilon threshold')
parser.add_argument('--max_steps_per_game', type=int, default=50000, help='Max steps per episode')
parser.add_argument('--start_learning', type=int, default=10000, help='Learning starts after these many steps')
parser.add_argument('--steps_to_take', type=int, default=100000, help='Total number of steps that the agent takes while training')

# setup parser arguments
args = parser.parse_args()

# load environment
env = atari_wrappers.wrap_deepmind(atari_wrappers.make_atari('SeaquestNoFrameskip-v4'),clip_rewards=False, frame_stack=True)
# get shape of actions
NUM_ACTIONS = env.action_space.n
# get shape of observations
OBS_SHAPE = env.observation_space.shape

# setup placeholders
input = tf.placeholder(tf.float32, [None, 84, 84, 4], name='input_placeholder')
expected_vals = tf.placeholder(tf.float32, [None, 84, 84, 4] )


# setup hyperparameters
TARGET_UPDATE_STEP_FREQ = args.target_update
BATCH_SIZE = args.batch_size
EPS_START = args.eps_start
EPS_END = args.eps_end
EPS_DECAY = args.eps_decay
EPISODE_NUM = args.ep_num
REPLAY_BUFFER_SIZE = 10000
STEPS_TO_TAKE = args.steps_to_take
start_learning=args.start_learning
max_steps_per_game = args.max_steps_per_game
LEARNING_RATE = args.learning_rate
GAMMA = 0.99

# setup policy and target model, replay memory, and optimizer
policy_model = AtariNet(input, OBS_SHAPE, NUM_ACTIONS, LEARNING_RATE)
target_model = AtariNet(input, OBS_SHAPE, NUM_ACTIONS, LEARNING_RATE)
replay_memory = ReplayMemory(REPLAY_BUFFER_SIZE)


print("Batch Size: {}".format(BATCH_SIZE))
print("Episodes: {}".format(EPISODE_NUM))
print("Target Update Freq: {}".format(TARGET_UPDATE_STEP_FREQ))
print("========================\n")
print("Epsilon Start: {}".format(EPS_START))
print("Epsilon End: {}".format(EPS_END))
print("Epsilon Decay: {}".format(EPS_DECAY))
print("========================\n")
print("Max steps per game: {}".format(max_steps_per_game))
print("Learning starts (after steps): {}".format(start_learning))
print("Total training steps: {}".format(STEPS_TO_TAKE))
print("========================\n")

with tf.Session() as session:
    # initialize variables
    session.run(tf.global_variables_initializer())
    # setup variable counters
    total_steps, exploit, explore, episode = 0, 0, 0, 0
    score_list = [] 

    # wait till atleast 64 observations are loaded into replay memory
    while len(replay_memory) < BATCH_SIZE:
        # get previous observeration
        prev_observation = env.reset()
        # get new observation based on random action
        cur_observation, cur_reward, done, _ = env.step(random.randrange(NUM_ACTIONS))
        # stack observations
        prepped_obs = np.expand_dims(np.array(cur_observation, dtype=np.float32), axis=0)
        # take greedy action
        action, count_explore, count_exploit = select_eps_greedy_action(session, input, policy_model, prepped_obs, total_steps, NUM_ACTIONS, EPS_START, EPS_END, EPS_DECAY, exploit, explore)

        next_observation, next_reward, done, info = env.step(action)
        # add to memory
        print("Filling Replay Memory.", end='\r')
        replay_memory.push(prev_observation, action, cur_observation, cur_reward, next_observation, next_reward)
    
    print("\n====================\n")
    print("Training Start\n")
    while total_steps < STEPS_TO_TAKE:
        
        episode+=1
        print("------------------")
        print("| Episode: {}".format(episode))
        # get previous observation
        prev_observation = env.reset()
        # take random action and get observations
        cur_observation, cur_reward, done, _ = env.step(random.randrange(NUM_ACTIONS))
        done = False
        # setup variables
        ep_score, steps, exploit, explore = 0, 0, 0, 0
        
        while not done: # until the episode ends
            steps += 1
            # print("| Steps: {}".format(steps), end='\r')
            # select a greedy action and get observations
            prepped_obs = np.expand_dims(np.array(cur_observation, dtype=np.float32), axis=0)
            
            action, count_explore, count_exploit = select_eps_greedy_action(session, input, policy_model, prepped_obs, total_steps, NUM_ACTIONS, EPS_START, EPS_END, EPS_DECAY, exploit, explore)
            
            next_observation, next_reward, done, info = env.step(action)
            # add to memory
            replay_memory.push(prev_observation, action, cur_observation, cur_reward, next_observation, next_reward)
            
            # set previous observation to current observation
            prev_observation = cur_observation
            cur_observation = next_observation

            # before enough transitions are collected to form a batch
            if len(replay_memory) < BATCH_SIZE:
                break

            # start learning after a certain number of steps    
            if total_steps > start_learning:
                # prepare training batch
                transitions = replay_memory.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                next_state_batch = np.array(batch.next_state, dtype=np.float32)
                cur_state_batch = np.array(batch.cur_state, dtype=np.float32)
                action_batch = np.array(batch.action, dtype=np.int64)
                cur_reward_batch = np.array(batch.cur_reward)
                next_reward_batch = np.array(batch.next_reward)

                # state values
                state_actions = session.run([policy_model.output], feed_dict={input: cur_state_batch})


                # calculate best value at next state
                next_state_actions = session.run([target_model.output], feed_dict={input:next_state_batch})
                next_state_values = np.amax(next_state_actions[0], axis=1)

                # compute the expected Q values

                # 2 step time difference
                target_qvals = cur_reward_batch + (GAMMA * next_reward_batch) + ((GAMMA**2) * next_state_values)

                # 1 step time difference
                # target_qvals = cur_reward_batch + (GAMMA * next_state_values)
                # optimize
                loss, _ = session.run([policy_model.loss, policy_model.train_op], feed_dict={input: cur_state_batch, policy_model.target_Q: target_qvals, policy_model.actions: state_actions[0]})
                
            
            # update variable values
            ep_score += next_reward
            total_steps += 1
            exploit= count_exploit 
            explore = count_explore 
            
            # if agent is not doing anything significant
            if steps >= 1000 and next_reward == 0.0:
                break

            
            
            #update the target network, copying all variables in DQN
            if total_steps % TARGET_UPDATE_STEP_FREQ == 0:

                print("------------Updating Target Model------------")
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
                # save target model
                target_model.saver.save(session, args.model_dir + "target/" + "homework_3")
                if total_steps == (STEPS_TO_TAKE//2):
                    target_model.saver.save(session, args.model_dir + "target_halfway/" + "homework_3")
                                    
        print("| Steps: {}".format(steps))
        print("| Score: {}".format(ep_score))
        print("| Explore: {}".format(count_explore))
        print("| Exploit: {}".format(count_exploit))
        print("| Total steps taken: {}".format(total_steps))
        print("------------------")
        score_list.append(ep_score)

    avg_score = (sum(score_list))/(len(score_list))
    print("\nAverage episode score: {}".format(avg_score))
    print("Loss: ".format(loss))
    print("Top score for all episodes: {}".format(max(score_list)))
    print("Total steps taken: {}".format(total_steps))
    policy_model.saver.save(session, args.model_dir + "policy/" + "homework_3")

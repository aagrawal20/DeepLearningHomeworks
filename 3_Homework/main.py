"""Module trains the agent"""
import argparse
import random
import numpy as np
import tensorflow as tf
import atari_wrappers
from util import select_eps_greedy_action, ReplayMemory, PrioritizedReplayBuffer
from model import AtariNet

# setup PARSER
PARSER = argparse.ArgumentParser(description='Train Atari Agent.')
PARSER.add_argument(
    '--model_dir',
    type=str,
    default='/work/cse496dl/teams/Dropouts/3_Homework/test_agent/',
    help='directory where model graph and WEIGHTS are saved')
PARSER.add_argument(
    '--BATCH_SIZE',
    type=int,
    default=32,
    help='Mini batch size for training')
PARSER.add_argument(
    '--LEARNING_RATE',
    type=float,
    default=0.0001,
    help='Learning rate for optimizer')
PARSER.add_argument(
    '--target_update',
    type=int,
    default=10,
    help='Frequency of Target update step')
PARSER.add_argument(
    '--EPS_START',
    type=float,
    default=1.)
PARSER.add_argument(
    '--EPS_END',
    type=float,
    default=0.01)
PARSER.add_argument(
    '--EPS_DECAY',
    type=int,
    default=100000,
    help='Decay for the epsilon threshold')
PARSER.add_argument(
    '--MAX_STEPS_PER_GAME',
    type=int,
    default=50000,
    help='Max STEPS per EPISODE')
PARSER.add_argument(
    '--START_LEARNING',
    type=int,
    default=10000,
    help='Learning starts after these many STEPS')
PARSER.add_argument(
    '--STEPS_TO_TAKE',
    type=int,
    default=100000,
    help='Total number of STEPS that the agent takes while training')

# setup PARSER arguments
ARGS = PARSER.parse_args()

# load environment
ENV = atari_wrappers.wrap_deepmind(atari_wrappers.make_atari('SeaquestNoFrameskip-v4'),
                                   clip_rewards=False, frame_stack=True)
# get shape of ACTIONs
NUM_ACTIONS = ENV.action_space.n
# get shape of observations
OBS_SHAPE = ENV.observation_space.shape

# setup placeholders
INPUT = tf.placeholder(tf.float32, [None, 84, 84, 4], name='input_placeholder')


# setup hyperparameters
TARGET_UPDATE_STEP_FREQ = ARGS.target_update
BATCH_SIZE = ARGS.batch_size
EPS_START = ARGS.eps_start
EPS_END = ARGS.eps_end
EPS_DECAY = ARGS.EPS_DECAY
REPLAY_BUFFER_SIZE = 10000
STEPS_TO_TAKE = ARGS.STEPS_to_take
START_LEARNING = ARGS.start_learning
MAX_STEPS_PER_GAME = ARGS.max_STEPS_per_game
LEARNING_RATE = ARGS.learning_rate
GAMMA = 0.99

# setup policy and target model, replay memory
POLICY_MODEL = AtariNet(INPUT, OBS_SHAPE, NUM_ACTIONS, LEARNING_RATE)
TARGET_MODEL = AtariNet(INPUT, OBS_SHAPE, NUM_ACTIONS, LEARNING_RATE)
REPLAY_MEMORY = ReplayMemory(REPLAY_BUFFER_SIZE)
PRB_MEMORY = PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE, 0.6)

print("Batch Size: {}".format(BATCH_SIZE))
print("Target Update Freq: {}".format(TARGET_UPDATE_STEP_FREQ))
print("========================\n")
print("Epsilon Start: {}".format(EPS_START))
print("Epsilon End: {}".format(EPS_END))
print("Epsilon Decay: {}".format(EPS_DECAY))
print("========================\n")
print("Max STEPS per game: {}".format(MAX_STEPS_PER_GAME))
print("Learning starts (after STEPS): {}".format(START_LEARNING))
print("Total training STEPS: {}".format(STEPS_TO_TAKE))
print("========================\n")

with tf.Session() as session:
    # initialize variables
    session.run(tf.global_variables_initializer())
    # setup variable counters
    TOTAL_STEPS, EXPLOIT, EXPLORE, EPISODE = 0, 0, 0, 0
    # list to store scores
    SCORE_LIST = [] 

    # wait till atleast 64 observations are loaded into replay memory
    while len(PRB_MEMORY) < BATCH_SIZE:
        # get s using environment
        PREV_OBSERVATION = ENV.reset()
        # get s' r' using random ACTION
        CUR_OBSERVATION, CUR_REWARD, DONE, _ = ENV.step(random.randrange(NUM_ACTIONS))
        # stack observations
        PREPPED_OBS = np.expand_dims(np.array(CUR_OBSERVATION, dtype=np.float32), axis=0)
        # take greedy ACTION
        ACTION, COUNT_EXPLORE, COUNT_EXPLOIT = select_eps_greedy_action(session, input, POLICY_MODEL, PREPPED_OBS, TOTAL_STEPS, NUM_ACTIONS, EPS_START, EPS_END, EPS_DECAY, EXPLOIT, EXPLORE)
        # get s" r" using greedy ACTION
        NEXT_OBSERVATION, NEXT_REWARD, DONE, INFO = ENV.step(ACTION)
        # add to memory
        print("Filling Replay Memory.", end='\r')
        PRB_MEMORY.push(PREV_OBSERVATION, ACTION, CUR_OBSERVATION, CUR_REWARD, NEXT_OBSERVATION, NEXT_REWARD)
    
    print("\n====================\n")
    print("Training Start\n")
    while TOTAL_STEPS < STEPS_TO_TAKE:
        EPISODE += 1
        print("------------------")
        print("| EPISODE: {}".format(EPISODE))
        # get previous observation
        PREV_OBSERVATION = ENV.reset()
        # take random ACTION and get observations
        CUR_OBSERVATION, CUR_REWARD, DONE, _ = ENV.step(random.randrange(NUM_ACTIONS))
        # set game end flag
        DONE = False
        # setup variables
        EP_SCORE, STEPS, EXPLOIT, EXPLORE = 0, 0, 0, 0
 
        while not DONE: # until the EPISODE ends
            # increment step counter
            STEPS += 1
            # select a greedy ACTION and get observations
            PREPPED_OBS = np.expand_dims(np.array(CUR_OBSERVATION, dtype=np.float32), axis=0)
            # get greedy ACTION
            ACTION, COUNT_EXPLORE, COUNT_EXPLOIT = select_eps_greedy_action(session, input, 
                                                                            POLICY_MODEL,
                                                                            PREPPED_OBS,
                                                                            TOTAL_STEPS,
                                                                            NUM_ACTIONS,
                                                                            EPS_START,
                                                                            EPS_END,
                                                                            EPS_DECAY,
                                                                            EXPLOIT,
                                                                            EXPLORE)
            # get observation and reward based on greedy ACTION
            NEXT_OBSERVATION, NEXT_REWARD, DONE, INFO = ENV.step(ACTION)
            # add to memory
            PRB_MEMORY.push(PREV_OBSERVATION, ACTION, CUR_OBSERVATION, CUR_REWARD,
                            NEXT_OBSERVATION, NEXT_REWARD)
            # set previous observation to current observation
            PREV_OBSERVATION = CUR_OBSERVATION
            # set current observation to next observation
            CUR_OBSERVATION = NEXT_OBSERVATION

            # check if enough transitions are collected to form a batch
            if len(PRB_MEMORY) < BATCH_SIZE:
                break

            # start learning after a certain number of STEPS    
            if TOTAL_STEPS > START_LEARNING:
                # prepare training batch
                _, ACTION_BATCH, CUR_STATE_BATCH, CUR_REWARD_BATCH, NEXT_STATE_BATCH, NEXT_REWARD_BATCH, WEIGHTS, BATCH_IDXS = PRB_MEMORY.sample(BATCH_SIZE, 0.4)
                # state values
                STATE_ACTIONS = session.run([POLICY_MODEL.output], feed_dict={input: CUR_STATE_BATCH})
                # calculate best value at next state
                NEXT_STATE_ACTIONS = session.run([TARGET_MODEL.output], feed_dict={input:NEXT_STATE_BATCH})
                NEXT_STATE_VALUES = np.amax(NEXT_STATE_ACTIONS[0], axis=1)
                # compute the expected Q values with 2 step time difference
                TARGET_Q_VALS = CUR_REWARD_BATCH + (GAMMA * NEXT_REWARD_BATCH) + ((GAMMA**2) * NEXT_STATE_VALUES)
                # optimize
                LOSS, _, Q, TARGET_Q = session.run([POLICY_MODEL.loss, POLICY_MODEL.train_op, POLICY_MODEL.q_val, POLICY_MODEL.target_q], feed_dict={input: CUR_STATE_BATCH, POLICY_MODEL.target_q: TARGET_Q_VALS, POLICY_MODEL.actions: STATE_ACTIONS[0]})
                # compute td error
                TD_ERROR = Q - TARGET_Q
                # get new priorities
                NEW_PRIORITIES = np.abs(TD_ERROR) + (1e-6)
                # update priorities for replay buffer
                PRB_MEMORY.update_priorities(BATCH_IDXS, NEW_PRIORITIES)
  
            # update variable values
            EP_SCORE += NEXT_REWARD
            TOTAL_STEPS += 1
            EXPLOIT = COUNT_EXPLOIT 
            EXPLORE = COUNT_EXPLORE 

            # if agent is not doing anything significant
            if STEPS >= 1000 and NEXT_REWARD == 0.0:
                break

            #update the target network, copying all variables in DQN
            if TOTAL_STEPS % TARGET_UPDATE_STEP_FREQ == 0:

                print("------------Updating Target Model------------")
                # get trainable variables
                TRAINABLE_VARS = tf.trainable_variables()
                # length of trainable variables
                TOTAL_VARS = len(TRAINABLE_VARS)
                # list to hold all operators
                OPS = []

                # iterate through policy model WEIGHTS
                POLICY_MODEL_WEIGHTS = TRAINABLE_VARS[0:TOTAL_VARS//2]
                for idx, var in enumerate(POLICY_MODEL_WEIGHTS):
                    # get target model WEIGHTS
                    TARGET_MODEL_WEIGHTS = TRAINABLE_VARS[idx + TOTAL_VARS//2]
                    # assign policy model WEIGHTS to target model WEIGHTS
                    OPS.append(TARGET_MODEL_WEIGHTS.assign((var.value())))

                # run session to transfer WEIGHTS
                for op in OPS:
                    session.run(op)
                # save target model
                TARGET_MODEL.saver.save(session, ARGS.model_dir + "target/" + "homework_3")
                # save the model halfway
                if TOTAL_STEPS == (STEPS_TO_TAKE//2):
                    TARGET_MODEL.saver.save(session, ARGS.model_dir + "target_halfway/" + "homework_3")
                                    
        print("| Steps: {}".format(STEPS))
        print("| Score: {}".format(EP_SCORE))
        print("| EXPLORE: {}".format(COUNT_EXPLORE))
        print("| EXPLOIT: {}".format(COUNT_EXPLOIT))
        print("| Total STEPS taken: {}".format(TOTAL_STEPS))
        print("------------------")
        SCORE_LIST.append(EP_SCORE)

    AVG_SCORE = (sum(SCORE_LIST))/(len(SCORE_LIST))
    print("\nAverage EPISODE score: {}".format(AVG_SCORE))
    print("Loss: ".format(LOSS))
    print("Top score for all EPISODEs: {}".format(max(SCORE_LIST)))
    print("Total STEPS taken: {}".format(TOTAL_STEPS))
    # save policy model
    POLICY_MODEL.saver.save(session, ARGS.model_dir + "policy/" + "homework_3")

#!/usr/bin/env python
from __future__ import print_function

import skimage as skimage
from skimage import transform, color, exposure
from skimage.viewer import ImageViewer
import random
from random import choice
import numpy as np
from collections import deque
import time

import json
from keras.models import model_from_json
from keras.models import Sequential, load_model, Model
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Dropout, Activation, Flatten, RepeatVector, Masking
from keras.layers import Convolution2D, Dense, Flatten, merge, MaxPooling2D, Input, AveragePooling2D, Lambda, Merge, Activation, Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import SGD, Adam, rmsprop
from keras import backend as K

from vizdoom import DoomGame, ScreenResolution
from vizdoom import *
import itertools as it
from time import sleep
import tensorflow as tf

from networks import Networks


def preprocessImg(img, size):

    img = np.rollaxis(img, 0, 3)    # It becomes (640, 480, 3)
    img = skimage.transform.resize(img,size)

    return img

class ReplayMemory():
    """
    Memory Replay Buffer 
    """

    def __init__(self, buffer_size=10000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, episode_experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1+len(self.buffer))-self.buffer_size] = []
        self.buffer.append(episode_experience)

    def sample(self, batch_size, trace_length):
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode)+1-trace_length)
            sampledTraces.append(episode[point:point+trace_length])
        sampledTraces = np.array(sampledTraces)
        return sampledTraces

class DoubleDQNAgent:

    def __init__(self, state_size, action_size, trace_length):

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the Double DQN
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.0001
        self.batch_size = 32
        self.observe = 5000
        self.explore = 50000
        self.frame_per_action = 4
        self.trace_length = trace_length
        self.update_target_freq = 3000
        self.timestep_per_train = 5 # Number of timesteps between training interval

        # Create replay memory
        self.memory = ReplayMemory()

        # Create main model and target model
        self.model = None
        self.target_model = None

        # Performance Statistics
        self.stats_window_size= 50 # window size for computing rolling statistics
        self.mavg_score = [] # Moving Average of Survival Time
        self.var_score = [] # Variance of Survival Time
        self.mavg_ammo_left = [] # Moving Average of Ammo used
        self.mavg_kill_counts = [] # Moving Average of Kill Counts

    def update_target_model(self):
        """
        After some time interval update the target model to be same with model
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        """
        Get action from model using epsilon-greedy policy
        """
        if np.random.rand() <= self.epsilon:
            action_idx = random.randrange(self.action_size)
        else:
  
            # Use all traces for RNN
            #q = self.model.predict(state) # 1x8x3
            #action_idx = np.argmax(q[0][-1])

            # Only use last trace for RNN
            q = self.model.predict(state) # 1x3
            action_idx = np.argmax(q)
        return action_idx

    def shape_reward(self, r_t, misc, prev_misc, t):
        
        # Check any kill count
        if (misc[0] > prev_misc[0]):
            r_t = r_t + 1

        if (misc[1] < prev_misc[1]): # Use ammo
            r_t = r_t - 0.1

        if (misc[2] < prev_misc[2]): # Loss HEALTH
            r_t = r_t - 0.1

        return r_t

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):

        sample_traces = self.memory.sample(self.batch_size, self.trace_length) # 32x8x4

        # Shape (batch_size, trace_length, img_rows, img_cols, color_channels)
        update_input = np.zeros(((self.batch_size,) + self.state_size)) # 32x8x64x64x3
        update_target = np.zeros(((self.batch_size,) + self.state_size))

        action = np.zeros((self.batch_size, self.trace_length)) # 32x8
        reward = np.zeros((self.batch_size, self.trace_length))

        for i in range(self.batch_size):
            for j in range(self.trace_length):
                update_input[i,j,:,:,:] = sample_traces[i][j][0]
                action[i,j] = sample_traces[i][j][1]
                reward[i,j] = sample_traces[i][j][2]
                update_target[i,j,:,:,:] = sample_traces[i][j][3]

        """
        # Use all traces for training
        # Size (batch_size, trace_length, action_size)
        target = self.model.predict(update_input) # 32x8x3
        target_val = self.model.predict(update_target) # 32x8x3

        for i in range(self.batch_size):
            for j in range(self.trace_length):
                a = np.argmax(target_val[i][j])
                target[i][j][int(action[i][j])] = reward[i][j] + self.gamma * (target_val[i][j][a])
        """

        # Only use the last trace for training
        target = self.model.predict(update_input) # 32x3
        target_val = self.model.predict(update_target) # 32x3

        for i in range(self.batch_size):
            a = np.argmax(target_val[i])
            target[i][int(action[i][-1])] = reward[i][-1] + self.gamma * (target_val[i][a])

        loss = self.model.train_on_batch(update_input, target)

        return np.max(target[-1,-1]), loss

    # load the saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":

    # Avoid Tensorflow eats up GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    game = DoomGame()
    game.load_config("../../scenarios/defend_the_center.cfg")
    game.set_sound_enabled(True)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(False)
    game.init()

    game.new_episode()
    game_state = game.get_state()
    misc = game_state.game_variables  # [KILLCOUNT, AMMO, HEALTH]
    prev_misc = misc

    action_size = game.get_available_buttons_size()

    img_rows , img_cols = 64, 64
    img_channels = 3 # Color channel
    trace_length = 4 # Temporal Dimension

    state_size = (trace_length, img_rows, img_cols, img_channels)
    agent = DoubleDQNAgent(state_size, action_size, trace_length)

    agent.model = Networks.drqn(state_size, action_size, agent.learning_rate)
    agent.target_model = Networks.drqn(state_size, action_size, agent.learning_rate)

    s_t = game_state.screen_buffer # 480 x 640
    s_t = preprocessImg(s_t, size=(img_rows, img_cols))

    is_terminated = game.is_episode_finished()

    # Start training
    epsilon = agent.initial_epsilon
    GAME = 0
    t = 0
    max_life = 0 # Maximum episode life (Proxy for agent performance)
    life = 0
    episode_buf = [] # Save entire episode

    # Buffer to compute rolling statistics 
    life_buffer, ammo_buffer, kills_buffer = [], [], [] 

    while not game.is_episode_finished():

        loss = 0
        Q_max = 0
        r_t = 0
        a_t = np.zeros([action_size])
        
        # Epsilon Greedy
        if len(episode_buf) > agent.trace_length:
            # 1x8x64x64x3
            state_series = np.array([trace[-1] for trace in episode_buf[-agent.trace_length:]])
            state_series = np.expand_dims(state_series, axis=0)
            action_idx  = agent.get_action(state_series)
        else:
            action_idx = random.randrange(agent.action_size)
        a_t[action_idx] = 1

        a_t = a_t.astype(int)
        game.set_action(a_t.tolist())
        skiprate = agent.frame_per_action
        game.advance_action(skiprate)

        game_state = game.get_state()  # Observe again after we take the action
        is_terminated = game.is_episode_finished()

        r_t = game.get_last_reward()  #each frame we get reward of 0.1, so 4 frames will be 0.4

        if (is_terminated):
            if (life > max_life):
                max_life = life
            GAME += 1
            life_buffer.append(life)
            ammo_buffer.append(misc[1])
            kills_buffer.append(misc[0])
            print ("Episode Finish ", misc)
            game.new_episode()
            game_state = game.get_state()
            misc = game_state.game_variables
            s_t1 = game_state.screen_buffer

        s_t1 = game_state.screen_buffer
        misc = game_state.game_variables
        s_t1 = preprocessImg(s_t1, size=(img_rows, img_cols))

        r_t = agent.shape_reward(r_t, misc, prev_misc, t)

        if (is_terminated):
            life = 0
        else:
            life += 1

        #update the cache
        prev_misc = misc

        # Update epsilon
        if agent.epsilon > agent.final_epsilon and t > agent.observe:
            agent.epsilon -= (agent.initial_epsilon - agent.final_epsilon) / agent.explore

        # Do the training
        if t > agent.observe:
            Q_max, loss = agent.train_replay()

        # save the sample <s, a, r, s'> to episode buffer
        episode_buf.append([s_t, action_idx, r_t, s_t1])

        if (is_terminated):
            agent.memory.add(episode_buf)
            episode_buf = [] # Reset Episode Buf

        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            print("Now we save model")
            agent.model.save_weights("models/drqn.h5", overwrite=True)

        # print info
        state = ""
        if t <= agent.observe:
            state = "observe"
        elif t > agent.observe and t <= agent.observe + agent.explore:
            state = "explore"
        else:
            state = "train"

        if (is_terminated):
            print("TIME", t, "/ GAME", GAME, "/ STATE", state, \
                  "/ EPSILON", agent.epsilon, "/ ACTION", action_idx, "/ REWARD", r_t, \
                  "/ Q_MAX %e" % np.max(Q_max), "/ LIFE", max_life, "/ LOSS", loss)

            # Save Agent's Performance Statistics
            if GAME % agent.stats_window_size == 0 and t > agent.observe: 
                print("Update Rolling Statistics")
                agent.mavg_score.append(np.mean(np.array(life_buffer)))
                agent.var_score.append(np.var(np.array(life_buffer)))
                agent.mavg_ammo_left.append(np.mean(np.array(ammo_buffer)))
                agent.mavg_kill_counts.append(np.mean(np.array(kills_buffer)))

                # Reset rolling stats buffer
                life_buffer, ammo_buffer, kills_buffer = [], [], [] 

                # Write Rolling Statistics to file
                with open("statistics/drqn_stats.txt", "w") as stats_file:
                    stats_file.write('Game: ' + str(GAME) + '\n')
                    stats_file.write('Max Score: ' + str(max_life) + '\n')
                    stats_file.write('mavg_score: ' + str(agent.mavg_score) + '\n')
                    stats_file.write('var_score: ' + str(agent.var_score) + '\n')
                    stats_file.write('mavg_ammo_left: ' + str(agent.mavg_ammo_left) + '\n')
                    stats_file.write('mavg_kill_counts: ' + str(agent.mavg_kill_counts) + '\n')


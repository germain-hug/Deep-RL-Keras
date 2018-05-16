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
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Dense, Flatten, merge, MaxPooling2D, Input, AveragePooling2D, Lambda, Merge, Activation, Embedding
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
    img = skimage.transform.resize(img, size)
    img = skimage.color.rgb2gray(img) 

    return img
    
class REINFORCEAgent:

    def __init__(self, state_size, action_size):
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.observe = 0
        self.frame_per_action = 4 # Frame skipping

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.learning_rate = 0.0001

        # Model for policy network
        self.model = None

        # Store episode states, actions and rewards
        self.states, self.actions, self.rewards = [], [], []

        # Performance Statistics
        self.stats_window_size= 50 # window size for computing rolling statistics
        self.mavg_score = [] # Moving Average of Survival Time
        self.var_score = [] # Variance of Survival Time
        self.mavg_ammo_left = [] # Moving Average of Ammo used
        self.mavg_kill_counts = [] # Moving Average of Kill Counts

    # Use the output of policy network, pick action stochastically (Stochastic Policy)
    def get_action(self, state):
        policy = self.model.predict(state).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0], policy

    # Instead agent uses sample returns for evaluating policy
    # Use TD(1) i.e. Monte Carlo updates 
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    # update policy network every episode
    def train_model(self):
        episode_length = len(self.states)

        discounted_rewards = self.discount_rewards(self.rewards)
        # Standardized discounted rewards
        discounted_rewards -= np.mean(discounted_rewards) 
        if np.std(discounted_rewards):
            discounted_rewards /= np.std(discounted_rewards)
        else:
            self.states, self.actions, self.rewards = [], [], []
            print ('std = 0!')
            return 0

        update_inputs = np.zeros(((episode_length,) + self.state_size)) # Episode_lengthx64x64x4
        # Similar to one-hot target but the "1" is replaced by discounted_rewards R_t
        advantages = np.zeros((episode_length, self.action_size))

        # Episode length is like the minibatch size in DQN
        for i in range(episode_length):
            update_inputs[i,:,:,:] = self.states[i]
            advantages[i][self.actions[i]] = discounted_rewards[i]
        
        loss = self.model.fit(update_inputs, advantages, nb_epoch=1, verbose=0)
        self.states, self.actions, self.rewards = [], [], []

        return loss.history['loss']


    def shape_reward(self, r_t, misc, prev_misc, t):
        
        # Check any kill count
        if (misc[0] > prev_misc[0]):
            r_t = r_t + 1

        if (misc[1] < prev_misc[1]): # Use ammo
            r_t = r_t - 0.1

        if (misc[2] < prev_misc[2]): # Loss HEALTH
            r_t = r_t - 0.1

        return r_t

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

    # Maximum number of episodes
    max_episodes = 1000000

    game.new_episode()
    game_state = game.get_state()
    misc = game_state.game_variables  # [KILLCOUNT, AMMO, HEALTH]
    prev_misc = misc

    action_size = game.get_available_buttons_size()

    img_rows , img_cols = 64, 64
    # Convert image into Black and white
    img_channels = 4 # We stack 4 frames

    state_size = (img_rows, img_cols, img_channels)
    agent = REINFORCEAgent(state_size, action_size)

    agent.model = Networks.policy_reinforce(state_size, action_size, agent.learning_rate)

    # Start training
    GAME = 0
    t = 0
    max_life = 0 # Maximum episode life (Proxy for agent performance)

    # Buffer to compute rolling statistics 
    life_buffer, ammo_buffer, kills_buffer = [], [], [] 

    for i in range(max_episodes):

        game.new_episode()
        game_state = game.get_state()
        misc = game_state.game_variables 
        prev_misc = misc

        x_t = game_state.screen_buffer # 480 x 640
        x_t = preprocessImg(x_t, size=(img_rows, img_cols))
        s_t = np.stack(([x_t]*4), axis=2) # It becomes 64x64x4
        s_t = np.expand_dims(s_t, axis=0) # 1x64x64x4

        life = 0 # Episode life

        while not game.is_episode_finished():

            loss = 0 # Training Loss at each update
            r_t = 0 # Initialize reward at time t
            a_t = np.zeros([action_size]) # Initialize action at time t

            x_t = game_state.screen_buffer
            x_t = preprocessImg(x_t, size=(img_rows, img_cols))
            x_t = np.reshape(x_t, (1, img_rows, img_cols, 1))
            s_t = np.append(x_t, s_t[:, :, :, :3], axis=3)
                
            # Sample action from stochastic softmax policy
            action_idx, policy  = agent.get_action(s_t)
            a_t[action_idx] = 1 

            a_t = a_t.astype(int)
            game.set_action(a_t.tolist())
            skiprate = agent.frame_per_action # Frame Skipping = 4
            game.advance_action(skiprate)

            r_t = game.get_last_reward()  # Each frame we get reward of 0.1, so 4 frames will be 0.4
            # Check if episode is terminated
            is_terminated = game.is_episode_finished()

            if (is_terminated):
                # Save max_life
                if (life > max_life):
                    max_life = life 
                life_buffer.append(life)
                ammo_buffer.append(misc[1])
                kills_buffer.append(misc[0])
                print ("Episode Finish ", prev_misc, policy)
            else:
                life += 1
                game_state = game.get_state()  # Observe again after we take the action
                misc = game_state.game_variables

            # Reward Shaping
            r_t = agent.shape_reward(r_t, misc, prev_misc, t)

            # Save trajactory sample <s, a, r> to the memory
            agent.append_sample(s_t, action_idx, r_t)

            # Update the cache
            t += 1
            prev_misc = misc

            if (is_terminated and t > agent.observe):
                # Every episode, agent learns from sample returns
                loss = agent.train_model()

            # Save model every 10000 iterations
            if t % 10000 == 0:
                print("Save model")
                agent.model.save_weights("models/reinforce.h5", overwrite=True)

            state = ""
            if t <= agent.observe:
                state = "Observe mode"
            else:
                state = "Train mode"

            if (is_terminated):

                # Print performance statistics at every episode end
                print("TIME", t, "/ GAME", GAME, "/ STATE", state, "/ ACTION", action_idx, "/ REWARD", r_t, "/ LIFE", max_life, "/ LOSS", loss)

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
                    with open("statistics/reinforce_stats.txt", "w") as stats_file:
                        stats_file.write('Game: ' + str(GAME) + '\n')
                        stats_file.write('Max Score: ' + str(max_life) + '\n')
                        stats_file.write('mavg_score: ' + str(agent.mavg_score) + '\n')
                        stats_file.write('var_score: ' + str(agent.var_score) + '\n')
                        stats_file.write('mavg_ammo_left: ' + str(agent.mavg_ammo_left) + '\n')
                        stats_file.write('mavg_kill_counts: ' + str(agent.mavg_kill_counts) + '\n')
        
        # Episode Finish. Increment game count
        GAME += 1

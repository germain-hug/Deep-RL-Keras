import gym
import numpy as np
from atari_wrappers import make_wrap_atari

"""
Atari Environment Helper Class
------
Original Code by
https://github.com/ShanHaoYu/Deep-Q-Network-Breakout/blob/master/environment.py
"""

class AtariEnvironment(object):
    def __init__(self, args, test=False):
        clip_rewards = not test
        self.env = make_wrap_atari(args.env, args.consecutive_frames, clip_rewards)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.consecutive_frames = args.consecutive_frames
        self.do_render = args.render

    def seed(self, seed):
        '''
        Control the randomness of the environment
        '''
        self.env.seed(seed)

    def reset(self):
        '''
        When running dqn:
            observation: np.array
                stack 4 last frames, shape: (84, 84, 4)
        When running pg:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)
        '''
        observation = self.env.reset()
        return np.array(observation)


    def step(self,action):
        '''
        When running dqn:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
            reward: int
                wrapper clips the reward to {-1, 0, 1} by its sign
                we don't clip the reward when testing
            done: bool
                whether reach the end of the episode?
        When running pg:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)
            reward: int
                if opponent wins, reward = +1 else -1
            done: bool
                whether reach the end of the episode?
        '''
        if not self.env.action_space.contains(action):
            raise ValueError('Ivalid action!!')

        if self.do_render:
            self.env.render()

        observation, reward, done, info = self.env.step(action)
        return np.array(observation), reward, done, info

    def get_action_size(self):
        return self.env.action_space.n

    def get_state_size(self):
        return 84, 84, self.consecutive_frames

    def get_random_action(self):
        return self.action_space.sample()

    def render(self):
        return self.env.render()

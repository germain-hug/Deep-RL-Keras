import gym
import numpy as np
from collections import deque

class Environment(object):
    """ Environment Helper Class (Multiple State Buffer) for Continuous Action Environments
    (MountainCarContinuous-v0, LunarLanderContinuous-v2, etc..), and MujuCo Environments
    """
    def __init__(self, gym_env, action_repeat):
        self.env = gym_env
        self.timespan = action_repeat
        self.gym_actions = 2 #range(gym_env.action_space.n)
        self.state_buffer = deque()

    def get_action_size(self):
        return self.env.action_space.n

    def get_state_size(self):
        return self.env.observation_space.shape

    def reset(self):
        """ Resets the game, clears the state buffer.
        """
        # Clear the state buffer
        self.state_buffer = deque()
        x_t = self.env.reset()
        s_t = np.stack([x_t for i in range(self.timespan)], axis=0)
        for i in range(self.timespan-1):
            self.state_buffer.append(x_t)
        return s_t

    def step(self, action):
        x_t1, r_t, terminal, info = self.env.step(action)
        previous_states = np.array(self.state_buffer)
        s_t1 = np.empty((self.timespan, *self.env.observation_space.shape))
        s_t1[:self.timespan-1, :] = previous_states
        s_t1[self.timespan-1] = x_t1
        # Pop the oldest frame, add the current frame to the queue
        self.state_buffer.popleft()
        self.state_buffer.append(x_t1)
        return s_t1, r_t, terminal, info

    def render(self):
        return self.env.render()

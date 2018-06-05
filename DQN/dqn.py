import sys
import random
import numpy as np

from agent import Agent
from random import random, randrange

sys.path.append('../utils/')
from memory_buffer import MemoryBuffer

class DQN:
    """ Deep Q-Learning Main Algorithm
    """

    def __init__(self, action_dim, state_dim, gamma = 0.99, epsilon = 1.0, epsilon_decay = 0.99, buffer_size = 200000, lr = 0.0001):
        """ Initialization
        """
        # Environment and DQN parameters
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        # Create actor and critic networks
        self.agent = Agent(state_dim, action_dim, lr)
        # Memory Buffer for Experience Replay
        self.buffer = MemoryBuffer(buffer_size)

    def policy_action(self, s):
        """ Apply an espilon-greedy policy
        """
        if random() <= self.epsilon:
            return randrange(self.action_dim)
        else:
            return np.argmax(self.agent.predict(s)[0])

    def train(self, batch_size):
        # Sample experience from memory buffer
        s, a, r, d, new_s = self.buffer.sample_batch(batch_size)
        # Apply Bellman Equation to train our DQN
        for i in range(s.shape[0]):
            new_r = r[i]
            if not d[i]: new_r = (r[i] + self.gamma * np.amax(self.agent.predict(new_s[i])[0]))
            target = self.agent.predict(s[i])
            target[0][a[i]] = new_r
            self.agent.fit(s[i], target)
        self.epsilon *= self.epsilon_decay

    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """
        self.buffer.memorize(state, action, reward, done, new_state)

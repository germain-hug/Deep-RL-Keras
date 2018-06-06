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

    def __init__(self, action_dim, state_dim, gamma = 0.99, epsilon = 0.25, epsilon_decay = 0.99, buffer_size = 1000000, lr = 0.00001, tau = 0.01):
        """ Initialization
        """
        # Environment and DQN parameters
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        # Create actor and critic networks
        self.agent = Agent(state_dim, action_dim, lr, tau)
        # Memory Buffer for Experience Replay
        self.buffer = MemoryBuffer(buffer_size)

    def policy_action(self, s):
        """ Apply an espilon-greedy policy to pick next action
        """
        if random() <= self.epsilon:
            return randrange(self.action_dim)
        else:
            return np.argmax(self.agent.predict(s)[0])

    def train(self, batch_size):
        """ Train on batch sampled from the buffer
        """
        # Sample experience from memory buffer
        s, a, r, d, new_s = self.buffer.sample_batch(batch_size)
        # Apply Bellman Equation on batch samples to train our DQN
        target = np.zeros((batch_size, self.action_dim))
        for i in range(s.shape[0]):
            new_r = r[i]
            if not d[i]: new_r = (r[i] + self.gamma * np.amax(self.agent.target_predict(new_s[i])[0]))
            q_value = self.agent.predict(s[i])[0]
            q_value[a[i]] = new_r
            target[i, :] = q_value
        # Train on batch
        self.agent.fit(s, target)
        # Decay epsilon
        self.epsilon *= self.epsilon_decay
        # Transfer weights to target network
        self.agent.transfer_weights()

    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """
        self.buffer.memorize(state, action, reward, done, new_state)

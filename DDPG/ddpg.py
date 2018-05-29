import numpy as np

from actor import Actor
from critic import Critic
from memory_buffer import MemoryBuffer

class DDPG:
    """ Deep Deterministic Policy Gradient (DDPG) Main Algorithm
    """

    def __init__(self, act_dim, env_dim, buffer_size = 100000, gamma = 0.99, lr = 0.001, tau=0.001):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.env_dim = env_dim
        # Create actor and critic networks
        self.actor = Actor(env_dim, act_dim, self.shared, lr, tau)
        self.critic = Critic(env_dim, act_dim, self.shared, lr, tau)
        self.buffer = MemoryBuffer(buffer_size)

    def policy_action(self, s):
        """ **OBSOLETE** Use the actor to predict the next action to take, using the policy
        """
        return np.random.choice(np.arange(self.act_dim), 1, p=self.actor.predict(s).ravel())[0]

    def train(self, states, actions, rewards, done):
        """ **OBSOLETE** Update actor and critic networks from experience
        """
        # Compute discounted rewards and Advantage (TD. Error)
        discounted_rewards = self.discount(rewards)
        state_values = self.critic.predict(np.array(states))
        advantages = discounted_rewards - np.reshape(state_values, len(state_values))
        # Networks optimization
        self.a_opt([states, actions, advantages])
        self.c_opt([states, discounted_rewards])

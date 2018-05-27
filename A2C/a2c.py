import random
import numpy as np

from keras.models import Model
from keras import regularizers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, SimpleRNN

from critic import Critic
from actor import Actor

class A2C:
    """ Actor-Critic Main Algorithm
    """

    def __init__(self, act_dim, env_dim, gamma = 0.99, lr = 0.001):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.gamma = gamma
        self.epsilon = 0.1
        # Create actor and critic networks
        self.shared = self.buildNetwork()
        self.actor = Actor(env_dim, act_dim, self.shared, lr)
        self.critic = Critic(env_dim, act_dim, self.shared, lr)
        # Build optimizers
        self.a_opt = self.actor.optimizer()
        self.c_opt = self.critic.optimizer()

    def buildNetwork(self):
        """ Assemble shared layers
        """
        inp = Input((self.env_dim))
        # If we have an image, apply convolutional layers
        if(len(self.env_dim) > 2):
            x = self.conv_block(inp, 32)
            x = self.conv_block(x, 32)
            x = Flatten()(x)
            x = Dense(32, activation='relu')(x)
        else:
            x = Dense(128, activation='relu')(inp)
            #x = SimpleRNN(128, activation='relu', dropout=0.2, recurrent_dropout=0.2)(inp)
        return Model(inp, x)

    def conv_layer(self, d):
        """ Returns a 2D Conv layer, with L2-regularization and ReLU activation
        """
        return Conv2D(d, 3,
            activation = 'relu',
            padding = 'same',
            kernel_initializer = 'he_normal')

    def conv_block(self, inp, d):
        """ Returns a 2D Conv block, with a convolutional layer, max-pooling,
        dropout and batch-normalization
        """
        conv = self.conv_layer(d)(inp)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        return pool

    def policy_action(self, s):
        """ Use the actor to predict the next action to take, using the policy
        """
        if random.random() > self.epsilon:
            self.epsilon *= 0.98
            return np.random.choice(np.arange(self.act_dim), 1)[0]
        else:
            return np.random.choice(np.arange(self.act_dim), 1, p=self.actor.predict(s).ravel())[0]

    def discount(self, r):
        """ Compute the gamma-discounted rewards over an episode
        """
        discounted_r, cumul_r = np.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

    def train(self, states, actions, rewards, done):
        """ Update actor and critic networks from experience
        """
        # Compute discounted rewards and Advantage (TD. Error)
        discounted_rewards = self.discount(rewards)
        state_values = self.critic.predict(np.array(states))
        advantages = discounted_rewards - np.reshape(state_values, len(state_values))
        # Networks optimization
        self.a_opt([states, actions, advantages])
        self.c_opt([states, discounted_rewards])

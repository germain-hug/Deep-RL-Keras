import numpy as np
import keras.backend as K

from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam
from .agent import Agent

class Critic(Agent):
    """ Critic for the A2C Algorithm
    """

    def __init__(self, inp_dim, out_dim, network, lr):
        Agent.__init__(self, inp_dim, out_dim, lr)
        self.model = self.addHead(network)
        self.discounted_r = K.placeholder(shape=(None,))

    def addHead(self, network):
        """ Assemble Critic network to predict value of each state
        """
        x = Dense(128, activation='relu')(network.output)
        out = Dense(1, activation='linear')(x)
        return Model(network.input, out)

    def optimizer(self):
        """ Critic Optimization: Mean Squared Error over discounted rewards
        """
        critic_loss = K.mean(K.square(self.discounted_r - self.model.output))
        updates = self.rms_optimizer.get_updates(self.model.trainable_weights, [], critic_loss)
        return K.function([self.model.input, self.discounted_r], [], updates=updates)

    def save(self, path):
        self.model.save_weights(path + '_critic.h5')

    def load_weights(self, path):
        self.model.load_weights(path)

import numpy as np

from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam
from agent import Agent

class Critic(Agent):
    """ Critic for the A2C Algorithm
    """

    def __init__(self, inp_dim, out_dim, network, lr):
        Agent.__init__(self, inp_dim, out_dim)
        self.model = self.addHead(network)
        self.model.compile(Adam(lr, decay=1e-6), 'mse')
        print(self.model.summary())

    def addHead(self, network):
        """ Assemble Actor network to predict probability of each action
        """
        out = Dense(1, activation='linear')(network.output)
        return Model(network.input, out)

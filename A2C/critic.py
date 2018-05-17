import numpy as np

from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from agent import Agent

class Critic(Agent):
    """ Critic for the A2C Algorithm
    """

    def __init__(self, inp_dim, out_dim, lr):
        Agent.__init__(self, inp_dim, out_dim)
        self.model.compile(Adam(lr), 'mse')

    def network(self):
        """ Initialize Critic network
        """
        inp = Input((1, self.inp_dim))
        x = Dense(128, activation='relu')(inp)
        out = Dense(1)(x)
        return Model(inp, out)

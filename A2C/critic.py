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
        """ Critic network to predict value of each state
        """
        inp = Input((1, self.inp_dim))
        x = Dense(64, activation='relu', kernel_initializer='he_uniform')(inp)
        out = Dense(1, activation='linear')(x)
        return Model(inp, out)

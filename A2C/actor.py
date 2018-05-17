import numpy as np

from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from agent import Agent

class Actor(Agent):
    """ Actor for the A2C Algorithm
    """

    def __init__(self, inp_dim, out_dim, lr):
        Agent.__init__(self, inp_dim, out_dim)
        self.model.compile(Adam(lr), 'categorical_crossentropy')

    def network(self):
        """ Initialize actor network
        """
        inp = Input((1, self.inp_dim))
        x = Dense(128, activation='relu')(inp)
        out = Dense(self.out_dim, activation='softmax')(x)
        return Model(inp, out)

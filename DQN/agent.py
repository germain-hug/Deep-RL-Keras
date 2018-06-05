import sys
import numpy as np

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Flatten

sys.path.append('../utils/')
from networks import conv_block

class Agent:
    """ Agent Class (Network) for DQN
    """

    def __init__(self, state_dim, action_dim, lr):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.network()
        self.model.compile(Adam(lr), 'mse')

    def network(self):
        """ Build Deep Q-Network
        """
        inp = Input((self.state_dim))
        x = conv_block(inp, 32, (2, 2))
        x = conv_block(x, 32, (2, 2))
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(self.action_dim, activation='sigmoid')(x)
        return Model(inp, x)

    def fit(self, inp, targ):
        """ Perform one epoch of training
        """
        self.model.fit(self.reshape(inp), targ, epochs=1, verbose=0)

    def predict(self, inp):
        """ Critic Value Prediction
        """
        return self.model.predict(self.reshape(inp))

    def reshape(self, x):
        if len(x.shape) < 4: return np.expand_dims(x, axis=0)
        else: return x

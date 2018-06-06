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

    def __init__(self, state_dim, action_dim, lr, tau):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        # Initialize Deep Q-Network
        self.model = self.network()
        self.model.compile(Adam(lr), 'mse')
        # Build target Q-Network
        self.target_model = self.network()
        self.target_model.compile(Adam(lr), 'mse')
        self.target_model.set_weights(self.model.get_weights())

    def network(self):
        """ Build Deep Q-Network
        """
        inp = Input((self.state_dim))
        x = conv_block(inp, 32, (4, 4), 8)
        x = conv_block(x, 64, (2, 2), 4)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(self.action_dim, activation='sigmoid')(x)
        return Model(inp, x)

    def transfer_weights(self):
        """ Transfer Weights from Model to Target at rate Tau
        """
        W = self.model.get_weights()
        tgt_W = self.target_model.get_weights()
        for i in range(len(W)):
            tgt_W[i] = self.tau * W[i] + (1 - self.tau) * tgt_W[i]
        self.target_model.set_weights(tgt_W)

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

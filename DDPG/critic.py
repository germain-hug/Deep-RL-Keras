import numpy as np
import keras.backend as K

from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.layers import Input, Dense, concatenate

class Critic:
    """ Critic for the DDPG Algorithm, Q-Value function approximator
    """

    def __init__(self, inp_dim, out_dim, lr, tau):
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.tau = tau
        self.lr = lr
        #
        self.model = self.network()
        self.target_model = self.network()
        self.action_grads = tf.gradients(self.model.output, self.model.input[1])
        #
        self.model.compile(Adam(self.lr), 'mse')
        self.target_model.compile(Adam(self.lr), 'mse')

    def network(self):
        """ Assemble Critic network to predict value of each state
        """
        state = Input((self.env_dim))
        action = Input((self.act_dim))
        x1 = Dense(128, activation='relu')(state)
        x2 = Dense(128, activation='relu')(action)
        x = concatenate([x1, x2])
        x = LSTM(256)(x)
        x = Dense(1, activation='linear')(x)
        out = K.multiply(x, self.act_range)
        return Model([state, action], out)

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in xrange(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.target_model.set_weights(target_W)

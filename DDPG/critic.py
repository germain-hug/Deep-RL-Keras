import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.layers import Input, Dense, concatenate, LSTM, Reshape, Lambda

class Critic:
    """ Critic for the DDPG Algorithm, Q-Value function approximator
    """

    def __init__(self, inp_dim, out_dim, lr, tau):
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.tau = tau
        self.lr = lr
        #
        self.model = self.network()
        self.target_model = self.network()
        self.action_grads = tf.gradients(self.model.output, self.model.input[1])
        #
        self.model.compile(Adam(self.lr), 'mse')
        self.target_model.compile(Adam(self.lr), 'mse')
        print(self.model.summary())

    def network(self):
        """ Assemble Critic network to predict q-values
        """
        state = Input((self.env_dim))
        action = Input((self.act_dim,))
        x1 = Dense(128, activation='relu')(state)
        x2 = Dense(128, activation='relu')(action)
        x = concatenate([x1, x2])
        x = Reshape((1, 256))(x)
        x = LSTM(256)(x)
        out = Dense(1, activation='linear')(x)
        return Model([state, action], out)

    def target_predict(self, inp):
        """ Predict Q-Values using the target network
        """
        return self.target_model.predict(inp)

    def train_on_batch(self, states, actions, critic_target):
        """ Train the critic network on batch of sampled experience
        """
        return self.model.train_on_batch([states, actions], critic_target)

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in xrange(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.target_model.set_weights(target_W)

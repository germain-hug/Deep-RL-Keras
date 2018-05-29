import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.models import Model, load_model
from keras.layers import Input, Dense

class Actor:
    """ Actor Network for the DDPG Algorithm
    """

    def __init__(self, inp_dim, out_dim, lr, tau):
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.tau = tau
        self.lr = lr
        self.model = self.network()
        self.target_model = self.network()
        self.action_gdts = K.placeholder(shape=(None, self.out_dim))

    def network(self):
        """ Actor Network for Policy function Approximation, using a tanh
        activation for continuous control
        """
        inp = Input((self.env_dim))
        x = Dense(128, activation='relu')(inp)
        x = Dense(256, activation='relu')(x)
        x = Reshape((1, 256))(x)
        x = LSTM(256)(x)
        x = Dense(self.act_dim, activation='tanh')(x)
        out = K.multiply(x, self.act_range)
        return Model(inp, out)

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in xrange(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.target_model.set_weights(target_W)

    def optimizer(self):
        """ Actor Optimization
        """
        self.params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -self.action_gdts)
        grads = zip(self.params_grad, self.model.trainable_weights)
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)

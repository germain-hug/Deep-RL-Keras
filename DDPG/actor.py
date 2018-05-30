import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.models import Model, load_model
from keras.layers import Input, Dense, Reshape, LSTM, Lambda, BatchNormalization, GaussianNoise

class Actor:
    """ Actor Network for the DDPG Algorithm
    """

    def __init__(self, inp_dim, out_dim, act_range, lr, tau):
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.act_range = act_range
        self.tau = tau
        self.lr = lr
        self.model = self.network()
        self.target_model = self.network()
        self.action_gdts = K.placeholder(shape=(None, self.act_dim))

    def network(self):
        """ Actor Network for Policy function Approximation, using a tanh
        activation for continuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
        """
        inp = Input((self.env_dim))
        #
        x = Dense(128, activation='relu')(inp)
        x = GaussianNoise(0.1)(x)
        x = BatchNormalization()(x)
        #
        x = Dense(256, activation='relu')(x)
        x = GaussianNoise(0.1)(x)
        x = BatchNormalization()(x)
        #
        x = Reshape((1, 256))(x)
        x = LSTM(256)(x)
        x = Dense(self.act_dim, activation='tanh')(x)
        #
        out = Lambda(lambda i: i * self.act_range)(x)
        return Model(inp, out)

    def predict(self, state):
        return self.model.predict(np.expand_dims(state, axis=0))

    def target_predict(self, inp):
        return self.target_model.predict(inp)

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

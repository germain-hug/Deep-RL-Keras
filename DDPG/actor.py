import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Input, Dense, Reshape, LSTM, Lambda, BatchNormalization, GaussianNoise, Flatten

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
        self.adam_optimizer = self.optimizer()

    def network(self):
        """ Actor Network for Policy function Approximation, using a tanh
        activation for continuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
        """
        inp = Input((self.env_dim))
        #
        x = Dense(256, activation='relu')(inp)
        x = GaussianNoise(1.0)(x)
        #
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = GaussianNoise(1.0)(x)
        #
        out = Dense(self.act_dim, activation='tanh', kernel_initializer=RandomUniform())(x)
        out = Lambda(lambda i: i * self.act_range)(out)
        #
        return Model(inp, out)

    def predict(self, state):
        """ Action prediction
        """
        return self.model.predict(np.expand_dims(state, axis=0))

    def target_predict(self, inp):
        """ Action prediction (target network)
        """
        return self.target_model.predict(inp)

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.target_model.set_weights(target_W)

    def train(self, states, actions, grads):
        """ Actor Training
        """
        self.adam_optimizer([states, grads])

    def optimizer(self):
        """ Actor Optimizer
        """
        action_gdts = K.placeholder(shape=(None, self.act_dim))
        params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.model.trainable_weights)
        return K.function([self.model.input, action_gdts], [tf.train.AdamOptimizer(self.lr).apply_gradients(grads)])

    def save(self, path):
        self.model.save_weights(path + '_actor.h5')

    def load_weights(self, path):
        self.model.load_weights(path)

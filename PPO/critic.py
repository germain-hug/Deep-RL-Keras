import numpy as np
import keras.backend as K

from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam
from .agent import Agent

class Critic(Agent):
    """ Critic for the A3C Algorithm
    """

    def __init__(self, inp_dim, out_dim, lr):
        Agent.__init__(self, inp_dim, out_dim, lr)
        # self.model = self.addHead(network)
        self.model = self.build_critic(inp_dim, lr)
        # self.discounted_r = K.placeholder(shape=(None,))
        # Pre-compile for threading
        # self.model._make_predict_function()

    def build_critic(self, env_dim, lr):
        HIDDEN_SIZE = 128
        NUM_LAYERS = 2
        state_input = Input(shape=(env_dim,))
        x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(HIDDEN_SIZE, activation='tanh')(x)

        out_value = Dense(1)(x)

        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=lr), loss='mse')
        model.summary()

        return model

    # def addHead(self, network):
    #     """ Assemble Critic network to predict value of each state
    #     """
    #     x = Dense(128, activation='relu')(network.output)
    #     out = Dense(1, activation='linear')(x)
    #     return Model(network.input, out)

    # def optimizer(self):
    #     """ Critic Optimization: Mean Squared Error over discounted rewards
    #     """
    #     critic_loss = K.mean(K.square(self.discounted_r - self.model.output))
    #     updates = self.rms_optimizer.get_updates(self.model.trainable_weights, [], critic_loss)
    #     return K.function([self.model.input, self.discounted_r], [], updates=updates)

    def save(self, path):
        self.model.save_weights(path + '_critic.h5')

    def load_weights(self, path):
        self.model.load_weights(path)

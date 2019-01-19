import sys
import numpy as np
import keras.backend as K

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, LSTM, Lambda
from keras.regularizers import l2
from utils.networks import conv_block

class Agent:
    """ Agent Class (Network) for DDQN
    """

    def __init__(self, state_dim, action_dim, lr, tau, dueling):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        self.dueling = dueling
        # Initialize Deep Q-Network
        self.model = self.network(dueling)
        self.model.compile(Adam(lr), 'mse')
        # Build target Q-Network
        self.target_model = self.network(dueling)
        self.target_model.compile(Adam(lr), 'mse')
        self.target_model.set_weights(self.model.get_weights())

    def huber_loss(self, y_true, y_pred):
        return K.mean(K.sqrt(1 + K.square(y_pred - y_true)) - 1, axis=-1)

    def network(self, dueling):
        """ Build Deep Q-Network
        """
        inp = Input((self.state_dim))

        # Determine whether we are dealing with an image input (Atari) or not
        if(len(self.state_dim) > 2):
            inp = Input((self.state_dim[1:]))
            x = conv_block(inp, 32, (2, 2), 8)
            x = conv_block(x, 64, (2, 2), 4)
            x = conv_block(x, 64, (2, 2), 3)
            x = Flatten()(x)
            x = Dense(256, activation='relu')(x)
        else:
            x = Flatten()(inp)
            x = Dense(64, activation='relu')(x)
            x = Dense(64, activation='relu')(x)

        if(dueling):
            # Have the network estimate the Advantage function as an intermediate layer
            x = Dense(self.action_dim + 1, activation='linear')(x)
            x = Lambda(lambda i: K.expand_dims(i[:,0],-1) + i[:,1:] - K.mean(i[:,1:], keepdims=True), output_shape=(self.action_dim,))(x)
        else:
            x = Dense(self.action_dim, activation='linear')(x)
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
        """ Q-Value Prediction
        """
        return self.model.predict(self.reshape(inp))

    def target_predict(self, inp):
        """ Q-Value Prediction (using target network)
        """
        return self.target_model.predict(self.reshape(inp))

    def reshape(self, x):
        if len(x.shape) < 4 and len(self.state_dim) > 2: return np.expand_dims(x, axis=0)
        elif len(x.shape) < 3: return np.expand_dims(x, axis=0)
        else: return x

    def save(self, path):
        if(self.dueling):
            path += '_dueling'
        self.model.save_weights(path + '.h5')

    def load_weights(self, path):
        self.model.load_weights(path)

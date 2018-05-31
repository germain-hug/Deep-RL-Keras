import numpy as np
from keras.optimizers import RMSprop

class Agent:
    """ Agent Generic Class
    """

    def __init__(self, inp_dim, out_dim, lr, tau = 0.001):
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.tau = tau
        self.rms_optimizer = RMSprop(lr=lr, epsilon=0.1, rho=0.99)

    def fit(self, inp, targ):
        """ Perform one epoch of training
        """
        self.model.fit(self.reshape(inp), targ, epochs=1, verbose=0)

    def predict(self, inp):
        """ Critic Value Prediction
        """
        return self.model.predict(self.reshape(inp))

    def predict_target(self, inp):
        """ Critic Value Prediction
        """
        return self.target_model.predict(self.reshape(inp))

    def reshape(self, x):
        if len(x.shape) < 4: return np.expand_dims(x, axis=0)
        else: return x

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            print(W[i], target_W[i])
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.target_model.set_weights(target_W)

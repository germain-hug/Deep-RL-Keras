import numpy as np

from actor import Actor
from critic import Critic
from memory_buffer import MemoryBuffer

class DDPG:
    """ Deep Deterministic Policy Gradient (DDPG) Helper Class
    """

    def __init__(self, act_dim, env_dim, buffer_size = 100000, gamma = 0.99, lr = 0.001, tau=0.001):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.gamma = gamma
        # Create actor and critic networks
        self.actor = Actor(env_dim, act_dim, lr, tau)
        self.critic = Critic(env_dim, act_dim, lr, tau)
        self.buffer = MemoryBuffer(buffer_size)

    def get_action(self, s):
        """ Use the actor to predict value
        """
        return self.actor.predict(s)

    def target_critic_predict(self, s, a):
        """ Predict Q-Values using the target network
        """
        return self.critic.target_predict([s, a])

    def target_actor_predict(self, s):
        """ Predict Actions using the target network
        """
        return self.actor.target_predict(s)

    def bellman(self, states, rewards, q_values, dones):
        """ Use the Bellman Equation to compute the critic target 
        """
        critic_target = np.asarray(states)
        for i in range(states.shape[0]):
            critic_target[k] = rewards[k] + self.gamma * q_values[k] * dones[k]
        return critic_target

    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """
        self.buffer.memorize(state, action, reward, dones, new_state)

    def train_and_update(self, states, actions, rewards, done):
        """ Update actor and critic networks from sampled experience
        """
        # Train critic
        critic.train_on_batch(states,actions,critic_target)
        # Train actor
        a_for_grad = actor.model.predict(states)
        grads = critic.gradients(states, a_for_grad) # TODO
        actor.train(states, grads)
        # Transfer weights to target networks
        actor.transfer_weights()
        critic.transfer_weights()

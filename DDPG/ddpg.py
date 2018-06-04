import numpy as np

from actor import Actor
from critic import Critic
from memory_buffer import MemoryBuffer

class DDPG:
    """ Deep Deterministic Policy Gradient (DDPG) Helper Class
    """

    def __init__(self, act_dim, env_dim, act_range, k, buffer_size = 200000, gamma = 0.99, lr = 0.001, tau = 0.001):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.env_dim = (k,) + env_dim
        self.gamma = gamma
        # Create actor and critic networks
        self.actor = Actor(self.env_dim, act_dim, act_range, 0.1 * lr, tau)
        self.critic = Critic(self.env_dim, act_dim, lr, tau)
        self.buffer = MemoryBuffer(buffer_size)

    def get_action(self, s):
        """ Use the actor to predict value
        """
        return self.actor.predict(s)[0]

    def target_critic_predict(self, s, a):
        """ Predict Q-Values using the target network
        """
        return self.critic.target_predict([s, a])

    def target_actor_predict(self, s):
        """ Predict Actions using the target network
        """
        return self.actor.target_predict(s)

    def bellman(self, rewards, q_values, dones):
        """ Use the Bellman Equation to compute the critic target
        """
        critic_target = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            critic_target[i] = rewards[i] + self.gamma * q_values[i] * dones[i]
        return critic_target

    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """
        self.buffer.memorize(state, action, reward, done, new_state)

    def sample_batch(self, batch_size):
        return self.buffer.sample_batch(batch_size)

    def train_and_update(self, states, actions, critic_target):
        """ Update actor and critic networks from sampled experience
        """
        # Train critic
        self.critic.train_on_batch(states, actions, critic_target)
        # Q-Value Gradients under Current Policy
        actions = self.actor.model.predict(states)
        grads = self.critic.gradients(states, actions)
        # Train actor
        self.actor.train(states, actions, np.array(grads).reshape((-1, self.act_dim)))
        # Transfer weights to target networks at rate Tau
        self.actor.transfer_weights()
        self.critic.transfer_weights()

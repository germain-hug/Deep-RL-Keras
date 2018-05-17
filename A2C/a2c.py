import numpy as np

from critic import Critic
from actor import Actor

class A2C:
    """ Actor-Critic Main Algorithm
    """

    def __init__(self, act_dim, env_dim, gamma = 0.99, lr = 0.001):
        """ Initialization
        """
        # Create actor and critic networks
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.gamma = gamma
        self.actor = Actor(env_dim, act_dim, lr)
        self.critic = Critic(env_dim, act_dim, 5 * lr)

    def policy_action(self, s):
        """ Use the actor to predict the next action to take, stochastically
        """
        if(len(s.shape) == 2): s = np.expand_dims(s, axis=0)
        return np.random.choice(np.arange(self.act_dim), 1, p=self.actor.predict(s).ravel())[0]

    def train(self, s_0, a, r, s_1, done):
        """ Update actor and critic networks from experience
        """
        # Estimate state values using critic
        V_0, V_1 = self.critic.predict(s_0), self.critic.predict(s_1)
        # Compute actor and critic training targets
        critic_t, actor_t = 0, np.zeros((1, self.act_dim))
        if done:
            critic_t, actor_t[0][a] = r, r - V_0
        else:
            critic_t = r + self.gamma * V_1  # New state value (Critic Update)
            actor_t[0][a] = r + self.gamma * V_1 - V_0 # Approx. TD Error (Actor Update)
        # Train Actor and Critic
        self.actor.fit(s_0, actor_t)
        self.critic.fit(s_0, np.full((1, 1), critic_t))

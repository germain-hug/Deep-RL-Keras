import sys
import gym
import time
import threading
import numpy as np

from tqdm import tqdm
from keras.models import Model
from keras import regularizers
from keras.layers import Input, Dense, Flatten

from .critic import Critic
from .actor import Actor
from .thread import training_thread
from utils.atari_environment import AtariEnvironment
from utils.networks import conv_block
from utils.stats import gather_stats

class A3C:
    """ Asynchronous Actor-Critic Main Algorithm
    """

    def __init__(self, act_dim, env_dim, gamma = 0.99, lr = 0.0001):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.gamma = gamma
        # Create actor and critic networks
        self.shared = self.buildNetwork()
        self.shared_target = self.buildNetwork()
        self.shared_target.set_weights(self.shared.get_weights())
        self.actor = Actor(env_dim, act_dim, self.shared, self.shared_target, lr)
        self.critic = Critic(env_dim, act_dim, self.shared, self.shared_target, lr)
        # Build optimizers
        self.a_opt = self.actor.optimizer()
        self.c_opt = self.critic.optimizer()

    def buildNetwork(self):
        """ Assemble shared layers
        """
        inp = Input((self.env_dim))
        # If we have an image, apply convolutional layers
        if(len(self.env_dim) > 2):
            x = conv_block(inp, 32, (2, 2))
            x = conv_block(x, 32, (2, 2))
            x = Flatten()(x)
        else:
            x = Dense(64, activation='relu')(inp)
            x = Dense(128, activation='relu')(x)
        return Model(inp, x)

    def policy_action(self, s):
        """ Use the actor's target network to predict the next action to take, using the policy
        """
        return np.random.choice(np.arange(self.act_dim), 1, p=self.actor.predict(s).ravel())[0]

    def discount(self, r, done, s):
        """ Compute the gamma-discounted rewards over an episode
        """
        discounted_r, cumul_r = np.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

    def train_models(self, states, actions, rewards, done):
        """ Update actor and critic networks from experience
        """
        # Compute discounted rewards and Advantage (TD. Error)
        discounted_rewards = self.discount(rewards, done, states[-1])
        state_values = self.critic.predict(np.array(states))
        advantages = discounted_rewards - np.reshape(state_values, len(state_values))
        # Networks optimization
        self.a_opt([states, actions, advantages])
        self.c_opt([states, discounted_rewards])
        # Transfer weights to target network
        # self.actor.transfer_weights()
        # self.critic.transfer_weights()

    def train(self, env, args, summary_writer):

        # Instantiate one environment per thread
        if(args.is_atari):
            envs = [AtariEnvironment(args) for i in range(args.n_threads)]
            state_dim = envs[0].get_state_size()
            action_dim = envs[0].get_action_size()
        else:
            envs = [gym.make(args.env) for i in range(args.n_threads)]
            state_dim = envs[0].observation_space.shape
            action_dim = envs[0].action_space.n

        # Create threads
        factor = 100.0 / (args.nb_episodes)
        tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")

        threads = [threading.Thread(
                target=training_thread,
                args=(self,
                    args.nb_episodes,
                    envs[i],
                    action_dim,
                    args.consecutive_frames,
                    args.training_interval,
                    summary_writer,
                    tqdm_e,
                    factor)) for i in range(args.n_threads)]

        for t in threads:
            t.start()
            time.sleep(1)
        [t.join() for t in threads]

        return None

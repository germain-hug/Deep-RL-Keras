import sys
import gym
import time
import threading
import numpy as np

from tqdm import tqdm
from keras.models import Model
from keras import regularizers
from keras.layers import Input, Dense, Flatten, Reshape

from .critic import Critic
from .actor import Actor
from .thread import training_thread
from utils.atari_environment import AtariEnvironment
from utils.continuous_environments import Environment
from utils.networks import conv_block
from utils.stats import gather_stats

class A3C:
    """ Asynchronous Actor-Critic Main Algorithm
    """

    def __init__(self, act_dim, env_dim, k, gamma = 0.99, lr = 0.0001, is_atari=False):
        """ Initialization
        """
        # Environment and A3C parameters
        self.act_dim = act_dim
        if(is_atari):
            self.env_dim = env_dim
        else:
            self.env_dim = (k,) + env_dim
        self.gamma = gamma
        self.lr = lr
        # Create actor and critic networks
        self.shared = self.buildNetwork()
        self.actor = Actor(self.env_dim, act_dim, self.shared, lr)
        self.critic = Critic(self.env_dim, act_dim, self.shared, lr)
        # Build optimizers
        self.a_opt = self.actor.optimizer()
        self.c_opt = self.critic.optimizer()

    def buildNetwork(self):
        """ Assemble shared layers
        """
        inp = Input((self.env_dim))
        # If we have an image, apply convolutional layers
        if(len(self.env_dim) > 2):
            # Images
            x = Reshape((self.env_dim[1], self.env_dim[2], -1))(inp)
            x = conv_block(x, 32, (2, 2))
            x = conv_block(x, 32, (2, 2))
            x = Flatten()(x)
        elif(len(self.env_dim)==2):
            # 2D Inputs
            x = Flatten()(inp)
            x = Dense(64, activation='relu')(x)
            x = Dense(128, activation='relu')(x)
        else:
            # 1D Inputs
            x = Dense(64, activation='relu')(inp)
            x = Dense(128, activation='relu')(x)
        return Model(inp, x)

    def policy_action(self, s):
        """ Use the actor's network to predict the next action to take, using the policy
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

    def train(self, env, args, summary_writer):

        # Instantiate one environment per thread
        if(args.is_atari):
            envs = [AtariEnvironment(args) for i in range(args.n_threads)]
            state_dim = envs[0].get_state_size()
            action_dim = envs[0].get_action_size()
        else:
            envs = [Environment(gym.make(args.env), args.consecutive_frames) for i in range(args.n_threads)]
            [e.reset() for e in envs]
            state_dim = envs[0].get_state_size()
            action_dim = gym.make(args.env).action_space.n

        # Create threads
        tqdm_e = tqdm(range(int(args.nb_episodes)), desc='Score', leave=True, unit=" episodes")

        threads = [threading.Thread(
                target=training_thread,
                daemon=True,
                args=(self,
                    args.nb_episodes,
                    envs[i],
                    action_dim,
                    args.training_interval,
                    summary_writer,
                    tqdm_e,
                    args.render)) for i in range(args.n_threads)]

        for t in threads:
            t.start()
            time.sleep(0.5)
        try:
            [t.join() for t in threads]
        except KeyboardInterrupt:
            print("Exiting all threads...")
        return None

    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)

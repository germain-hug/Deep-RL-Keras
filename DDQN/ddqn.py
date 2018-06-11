import sys
import random
import numpy as np

from tqdm import tqdm
from .agent import Agent
from random import random, randrange

from utils.memory_buffer import MemoryBuffer
from utils.networks import tfSummary
from utils.stats import gather_stats

class DDQN:
    """ Deep Q-Learning Main Algorithm
    """

    def __init__(self, action_dim, state_dim, gamma = 0.99, epsilon = 0.25, epsilon_decay = 0.99, buffer_size = 100000, lr = 0.001, tau = 0.01):
        """ Initialization
        """
        # Environment and DDQN parameters
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        # Create actor and critic networks
        self.agent = Agent(state_dim, action_dim, lr, tau)
        # Memory Buffer for Experience Replay
        self.buffer = MemoryBuffer(buffer_size)

    def policy_action(self, s):
        """ Apply an espilon-greedy policy to pick next action
        """
        if random() <= self.epsilon:
            return randrange(self.action_dim)
        else:
            return np.argmax(self.agent.predict(s)[0])

    def train_agent(self, batch_size):
        """ Train Q-network on batch sampled from the buffer
        """
        # Sample experience from memory buffer
        s, a, r, d, new_s = self.buffer.sample_batch(batch_size)
        # Apply Bellman Equation on batch samples to train our DDQN
        target = np.zeros((batch_size, self.action_dim))
        for i in range(s.shape[0]):
            new_r = r[i]
            if not d[i]: new_r = (r[i] + self.gamma * np.amax(self.agent.target_predict(new_s[i])[0]))
            q_value = self.agent.predict(s[i])[0]
            q_value[a[i]] = new_r
            target[i, :] = q_value
        # Train on batch
        self.agent.fit(s, target)
        # Decay epsilon
        self.epsilon *= self.epsilon_decay
        # Transfer weights to target network
        self.agent.transfer_weights()

    def train(self, env, args, summary_writer):
        """ Main DDQN Training Algorithm
        """

        results = []
        tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")

        for e in tqdm_e:
            # Reset episode
            time, cumul_reward, done = 0, 0, False
            old_state = env.reset()
            actions, states, rewards = [], [], []

            while not done:
                if args.render: env.render()
                # Actor picks an action (following the policy)
                a = self.policy_action(old_state)
                # Retrieve new state, reward, and whether the state is terminal
                new_state, r, done, _ = env.step(a)
                # Memorize for experience replay
                self.memorize(old_state, a, r, done, new_state)
                # Update current state
                old_state = new_state
                cumul_reward += r
                time += 1

            # Train DDQN
            if(self.buffer.size() > args.batch_size):
                self.train_agent(args.batch_size)

            # Gather stats every 50 episode for plotting
            if(args.gather_stats):
                mean, stdev = gather_stats(self, env)
                results.append([e, mean, stdev])

            # Export results for Tensorboard
            score = tfSummary('score', cumul_reward)
            summary_writer.add_summary(score, global_step=e)
            summary_writer.flush()

            # Display score
            tqdm_e.set_description("Score: " + str(cumul_reward))
            tqdm_e.refresh()

        return results

    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """
        self.buffer.memorize(state, action, reward, done, new_state)

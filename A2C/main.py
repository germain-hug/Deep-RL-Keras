""" Advantage Actor-Critic Algorithm (A2C) for OpenAI Gym environment
"""

import sys
import gym
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from a2c import A2C
from tqdm import tqdm
from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical

sys.path.append('../utils/')
from networks import get_session, tfSummary

def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    #
    parser.add_argument('--nb_episodes', type=int, default=5000, help="Number of training episodes")
    parser.add_argument('--render', dest='render', action='store_true', help="Render environment")
    parser.add_argument('--env', type=str, default='CartPole-v1',help="OpenAI Gym Environment")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.set_defaults(render=False)
    return parser.parse_args(args)

def gather_stats(a2c, env):
    """ Compute average rewards over 10 episodes
    """
    score = []
    for k in range(10):
        old_state = env.reset()
        cumul_r, done = 0, False
        while not done:
            a = a2c.policy_action(old_state)
            old_state, r, done, _ = env.step(a)
            cumul_r += r
        score.append(cumul_r)
    return np.mean(np.array(score)), np.std(np.array(score))

def main(args=None):

    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Check if a GPU ID was set
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_session(get_session())
    summary_writer = tf.summary.FileWriter("./tensorboard_" + args.env)

    # Initialization
    env = gym.make(args.env)
    env_dim = env.observation_space.shape
    act_dim = env.action_space.n
    a2c = A2C(act_dim, env_dim)
    results = []

    # Main Loop
    tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")
    for e in tqdm_e:

        # Reset episode
        time, cumul_reward, done = 0, 0, False
        old_state = env.reset()
        actions, states, rewards = [], [], []

        while not done:
            if args.render: env.render()
            # Actor picks an action (following the policy)
            a = a2c.policy_action(old_state)
            # Retrieve new state, reward, and whether the state is terminal
            new_state, r, done, _ = env.step(a)
            # Memorize (s, a, r) for training
            actions.append(to_categorical(a, act_dim))
            rewards.append(r)
            states.append(old_state)
            # Update current state
            old_state = new_state
            cumul_reward += r
            time += 1

        # Train using discounted rewards ie. compute updates
        a2c.train(states, actions, rewards, done)
        # Gather stats every 50 episode for plotting
        if(e%50==0):
            mean, stdev = gather_stats(a2c, env)
            results.append([e, mean, stdev])
        # Export results for Tensorboard
        score = tfSummary('score', cumul_reward)
        summary_writer.add_summary(score, global_step=e)
        summary_writer.flush()
        # Display score
        tqdm_e.set_description("Score: " + str(cumul_reward))
        tqdm_e.refresh()

    # Export results to CSV
    df = pd.DataFrame(np.array(results))
    df.to_csv("logs.csv", header=['Episode', 'Mean', 'Stddev'], float_format='%10.5f')

    # Display agent
    while True:
        env.render()
        a = a2c.policy_action(old_state)
        old_state, r, done, _ = env.step(a)
        time += 1
        if done: env.reset()


if __name__ == "__main__":
    main()

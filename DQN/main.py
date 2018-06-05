""" Deep Q-Learning for OpenAI Gym environment
"""

import sys
import gym
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from dqn import DQN
from tqdm import tqdm
from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical

sys.path.append('../utils/')
from atari_environment import AtariEnvironment
from networks import get_session, tfSummary

def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    #
    parser.add_argument('--nb_episodes', type=int, default=5000, help="Number of training episodes")
    parser.add_argument('--render', dest='render', action='store_true', help="Render environment")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (experience replay)")
    parser.add_argument('--consecutive_frames', type=int, default=4, help="Number of consecutive frames (action repeat)")
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4',help="OpenAI Gym Environment")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.set_defaults(render=False)
    return parser.parse_args(args)

def gather_stats(dqn, env):
    """ Compute average rewards over 10 episodes
    """
    score = []
    for k in range(10):
        old_state = env.reset()
        cumul_r, done = 0, False
        while not done:
            a = dqn.policy_action(old_state)
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
    env = AtariEnvironment(args)
    state_dim = env.get_state_size()
    action_dim = env.get_action_size()
    dqn = DQN(action_dim, state_dim)
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
            a = dqn.policy_action(old_state)
            # Retrieve new state, reward, and whether the state is terminal
            new_state, r, done, _ = env.step(a)
            # Memorize for experience replay
            dqn.memorize(old_state, a, r, done, new_state)
            # Update current state
            old_state = new_state
            cumul_reward += r
            time += 1

        # Train DQN
        dqn.train(args.batch_size)

        # Gather stats every 50 episode for plotting
        if(e%50==0):
            mean, stdev = gather_stats(dqn, env)
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
        a = dqn.policy_action(old_state)
        old_state, r, done, _ = env.step(a)
        time += 1
        if done: env.reset()


if __name__ == "__main__":
    main()

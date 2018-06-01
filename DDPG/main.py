""" Deep Deterministic Policy Gradient (DDPG) for OpenAI Gym environment
"""

import os
import sys
import gym
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from keras.backend.tensorflow_backend import set_session

from ddpg import DDPG

episode = 0
gym.logger.set_level(40)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_session():
    """ Limit session memory usage
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.InteractiveSession(config=config)

def tfSummary(tag, val):
    """ Scalar Value Tensorflow Summary
    """
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])

def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    #
    parser.add_argument('--nb_episodes', type=int, default=5000, help="Number of training episodes")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (experience replay)")
    parser.add_argument('--gather_stats', dest='gather_stats', action='store_true',help="Compute Average reward per episode (slower)")
    parser.add_argument('--render', dest='render', action='store_true', help="Render environment")
    parser.add_argument('--env', type=str, default='BipedalWalker-v2',help="OpenAI Gym Environment")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.set_defaults(render=False)
    return parser.parse_args(args)

def gather_stats(agent, env):
    """ Compute average rewards over 10 episodes
    """
    score = []
    for k in range(10):
        old_state = env.reset()
        cumul_r, done = 0, False
        while not done:
            a = agent.get_action(old_state)
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
    env.reset()
    env_dim = env.observation_space.shape
    #print(env.action_space)
    act_dim, act_range = 2, 0.99 #env.action_space.n
    ddpg = DDPG(act_dim, env_dim, act_range)
    results = []

    # First, gather experience
    tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")
    for e in tqdm_e:

        # Reset episode
        time, cumul_reward, done = 0, 0, False
        old_state = env.reset()
        actions, states, rewards = [], [], []

        while not done:
            if args.render: env.render()
            # Actor picks an action (following the deterministic policy)
            a = ddpg.get_action(old_state)
            # Retrieve new state, reward, and whether the state is terminal
            new_state, r, done, _ = env.step(a)
            # Add outputs to memory buffer
            ddpg.memorize(old_state, a, r, done, new_state)
            # Sample experience from buffer
            states, actions, rewards, dones, new_states = ddpg.sample_batch(args.batch_size)
            # Predict target q-values using target networks
            q_values = ddpg.target_critic_predict(new_states, ddpg.target_actor_predict(new_states))
            # Compute critic target
            critic_target = ddpg.bellman(rewards, q_values, dones)
            # Train both networks on sampled batch, update target networks
            ddpg.train_and_update(states, actions, critic_target)
            # Gather stats every episode for plotting
            if(args.gather_stats):
                mean, stdev = gather_stats(ddpg, env)
                results.append([e, mean, stdev])
            # Update current state
            old_state = new_state
            cumul_reward += r
            time += 1


        # Export results for Tensorboard
        score = tfSummary('score', cumul_reward)
        summary_writer.add_summary(score, global_step=e)
        summary_writer.flush()
        # Display score
        tqdm_e.set_description("Score: " + str(cumul_reward))
        tqdm_e.refresh()

    # Export results to CSV
    if(args.gather_stats):
        df = pd.DataFrame(np.array(results))
        df.to_csv("logs.csv", header=['Episode', 'Mean', 'Stddev'], float_format='%10.5f')

    # Display agent
    while True:
        env.render()
        a = ddpg.get_action(old_state)
        old_state, r, done, _ = env.step(a)
        time += 1
        if done: env.reset()

if __name__ == "__main__":
    main()

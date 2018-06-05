""" Asynchronous Advantage Actor-Critic Algorithm (A3C) for OpenAI Gym environment
"""

import os
import sys
import gym
import argparse
import threading
import numpy as np
import tensorflow as tf
import time

from a3c import A3C
from tqdm import tqdm
from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical
from keras import backend as K

sys.path.append('../utils/')
from atari_environment import AtariEnvironment
from networks import get_session, tfSummary

episode = 0
gym.logger.set_level(40)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    #
    parser.add_argument('--n_threads', type=int, default=8, help="Number of threads")
    parser.add_argument('--nb_episodes', type=int, default=5000, help="Number of training episodes")
    parser.add_argument('--consecutive_frames', type=int, default=4, help="Number of consecutive frames (action repeat)")
    parser.add_argument('--training_interval', type=int, default=30, help="Network training frequency")
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4',help="OpenAI Gym Environment")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.set_defaults(render=False)
    return parser.parse_args(args)

def training_thread(agent, Nmax, env, action_dim, k, f, summary_writer, tqdm, factor):
    """ Build threads to run shared computation across
    """

    global episode
    while episode < Nmax:

        # Reset episode
        time, cumul_reward, done = 0, 0, False
        old_state = env.reset()
        actions, states, rewards = [], [], []
        while not done:
            # Actor picks an action (following the policy)
            a = agent.policy_action(np.expand_dims(old_state, axis=0))
            # Retrieve new state, reward, and whether the state is terminal
            new_state, r, done, _ = env.step(a)
            # Memorize (s, a, r) for training
            actions.append(to_categorical(a, action_dim))
            rewards.append(r)
            states.append(old_state)
            # Update current state
            old_state = new_state
            cumul_reward += r
            time += 1
            # Asynchronous training
            if(time%f==0 or done):
                agent.train(states, actions, rewards, done)
                actions, states, rewards = [], [], []

        # Export results for Tensorboard
        score = tfSummary('score', cumul_reward)
        summary_writer.add_summary(score, global_step=episode)
        summary_writer.flush()
        #
        tqdm.set_description("Score: " + str(cumul_reward))
        tqdm.update(int(episode * factor))
        episode += 1

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

    # Use AtariEnvironment Helper Class
    envs = [AtariEnvironment(args) for i in range(args.n_threads)]
    state_dim = envs[0].get_state_size()
    action_dim = envs[0].get_action_size()

    # Construct A3C Agent
    a3c = A3C(action_dim, state_dim)

    # Create threads
    factor = 100.0 / (args.nb_episodes)
    tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")

    threads = [threading.Thread(
            target=training_thread,
            args=(a3c,
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

if __name__ == "__main__":
    main()

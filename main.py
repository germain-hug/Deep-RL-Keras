""" Deep Q-Learning for OpenAI Gym environment
"""

import sys
import gym
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from DQN.dqn import DQN
from A2C.a2c import A2C

from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical

from utils.atari_environment import AtariEnvironment
from utils.networks import get_session

def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    #
    parser.add_argument('--type', type=str, default='DQN',help="Algorithm to train from {A2C, A3C, DQN, DDPG}")
    parser.add_argument('--is_atari', dest='is_atari', action='store_true', help="Atari Environment")
    #
    parser.add_argument('--nb_episodes', type=int, default=5000, help="Number of training episodes")
    parser.add_argument('--render', dest='render', action='store_true', help="Render environment while training")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (experience replay)")
    parser.add_argument('--consecutive_frames', type=int, default=4, help="Number of consecutive frames (action repeat)")
    #
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4',help="OpenAI Gym Environment")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    #
    parser.set_defaults(render=False)
    return parser.parse_args(args)

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

    # Environment Initialization
    if(args.is_atari):
        env = AtariEnvironment(args)
        state_dim = env.get_state_size()
        action_dim = env.get_action_size()
    else:
        env = gym.make(args.env)
        state_dim = env.observation_space.shape
        action_dim = env.action_space.n

    # Pick algorithm to train
    if(args.type=="DQN"):
        algo = DQN(action_dim, state_dim)
    elif(args.type=="A2C"):
        algo = A2C(action_dim, state_dim)

    # Train
    stats = algo.train(env, args, summary_writer)

    # Export results to CSV
    df = pd.DataFrame(np.array(stats))
    df.to_csv("logs.csv", header=['Episode', 'Mean', 'Stddev'], float_format='%10.5f')

    # Display agent
    old_state, time = env.reset(), 0
    while True:
        env.render()
        a = algo.policy_action(old_state)
        old_state, r, done, _ = env.step(a)
        time += 1
        if done: env.reset()


if __name__ == "__main__":
    main()

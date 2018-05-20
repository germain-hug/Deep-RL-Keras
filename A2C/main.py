""" Advantage Actor-Critic Algorithm (A2C) for OpenAI Gym environment
"""

import sys
import gym
import argparse
import numpy as np
import tensorflow as tf

from a2c import A2C
from tqdm import tqdm
from keras.backend.tensorflow_backend import set_session

def get_session():
    """ Limit session memory usage
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

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
    parser.add_argument('--render', dest='render', action='store_true', help="Render environment")
    parser.add_argument('--env', type=str, default='CartPole-v1',help="OpenAI Gym Environment")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
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

    # Initialization
    env = gym.make(args.env)
    env_dim  = env.observation_space.shape
    act_dim  = env.action_space.n
    print(env.observation_space.shape, env.action_space.n)

    a2c = A2C(act_dim, env_dim)
    solved = False

    # Main Loop
    for e in tqdm(range(args.nb_episodes)):

        time, cumul_reward, done = 0, 0, False
        old_state = env.reset()

        while not done:
            if args.render: env.render()
            # Actor picks an action (following the policy)
            a = a2c.policy_action(old_state)
            # Retrieve new state, reward, and whether the state is terminal
            new_state, r, done, _ = env.step(a)
            # Train ie. compute updates, updated both networks
            if not solved:
                a2c.train(old_state, a, r, new_state, done)
            # Update current state
            old_state = new_state
            cumul_reward += r
            time += 1

        # Export results for Tensorboard
        score = tfSummary('score', cumul_reward)
        summary_writer.add_summary(score, global_step=e)
        summary_writer.flush()

    while True:
        env.render()
        a = a2c.policy_action(old_state)
        old_state, r, done, _ = env.step(a)
        time += 1
        if done: env.reset()


if __name__ == "__main__":
    main()

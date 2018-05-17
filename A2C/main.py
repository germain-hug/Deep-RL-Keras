""" Actor-Critic Algorithm for OpenAI Gym environment
"""

import sys
import gym
import argparse
import numpy as np
import tensorflow as tf

from a2c import A2C
from keras.backend.tensorflow_backend import set_session

def get_session():
    """ Limit session memory usage
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

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

    # Initialization
    env = gym.make(args.env)
    env_dim  = env.observation_space.shape[0]
    act_dim  = env.action_space.n
    a2c = A2C(act_dim, env_dim)
    solved = False

    # Main Loop
    for e in range(args.nb_episodes):

        t, done = 0, False
        old_state = np.reshape(env.reset(), (1, env_dim))

        while not done:
            t += 1
            if args.render: env.render()
            # Actor picks an action (following the policy)
            a = a2c.policy_action(old_state)
            # Retrieve new state, reward, and whether we're done
            new_state, r, done, _ = env.step(a)
            # Reshape new state to match feeding to policy (expand_dims on axis 0)
            new_state = np.reshape(new_state, (1, env_dim))
            # (Optionally) Penalize for having failed
            if done and t < 500:
                r = -10
            # Train ie. compute updates, updated both networks
            if not solved:
                a2c.train(old_state, a, r, new_state, done and t < 500)
            # Update current state
            old_state = new_state

        if(e > 3000):
            args.render = True
        print('Episode %s lasted for %s steps' % (e, t))

if __name__ == "__main__":
    main()

""" Training thread for A3C
"""

import numpy as np
from threading import Thread, Lock
from keras.utils import to_categorical
from utils.networks import tfSummary

episode = 0
lock = Lock()

def training_thread(agent, Nmax, env, action_dim, f, summary_writer, tqdm, render):
    """ Build threads to run shared computation across
    """

    global episode
    while episode < Nmax:

        # Reset episode
        time, cumul_reward, done = 0, 0, False
        old_state = env.reset()
        actions, states, rewards = [], [], []
        while not done and episode < Nmax:
            if render:
                with lock: env.render()
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
                lock.acquire()
                agent.train_models(states, actions, rewards, done)
                lock.release()
                actions, states, rewards = [], [], []

        # Export results for Tensorboard
        score = tfSummary('score', cumul_reward)
        summary_writer.add_summary(score, global_step=episode)
        summary_writer.flush()
        # Update episode count
        with lock:
            tqdm.set_description("Score: " + str(cumul_reward))
            tqdm.update(1)
            if(episode < Nmax):
                episode += 1

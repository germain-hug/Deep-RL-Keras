import random
import numpy as np

from tqdm import tqdm
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers
from keras.utils import to_categorical
from keras.layers import Input, Dense, Flatten

from .critic import Critic
from .actor import Actor
from utils.networks import tfSummary
from utils.stats import gather_stats
from keras import backend as K

class PPO:
    """ Actor-Critic Main Algorithm
    """

    def __init__(self, act_dim, env_dim, k, gamma = 0.99, lr = 0.0001, loss_clipping=0.2, noise=1.0, entropy_loss=5e-3):
        """ Initialization
        """
        # PPO Params
        self.loss_clipping = loss_clipping
        self.noise = noise
        self.entropy_loss = entropy_loss

        # Environment and A2C parameters
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.gamma = gamma
        self.lr = lr

        # # Create actor and critic networks
        # self.shared = self.buildNetwork()
        # self.actor = Actor(self.env_dim, act_dim, self.shared, lr)
        # self.critic = Critic(self.env_dim, act_dim, self.shared, lr)

        # # Build optimizers
        # self.a_opt = self.actor.optimizer()
        # self.c_opt = self.critic.optimizer()


        self.actor = self.build_actor()
        self.critic = self.build_critic()

        self.rewards = []
        self.actor_losses = []
        self.critic_losses = []

        self.observation = None
        self.val = False

    # def buildNetwork(self):
    #     """ Assemble shared layers
    #     """
    #     inp = Input((self.env_dim))
    #     x = Flatten()(inp)
    #     x = Dense(64, activation='relu')(x)
    #     x = Dense(128, activation='relu')(x)
    #     return Model(inp, x)

    # def policy_action(self, s):
    #     """ Use the actor to predict the next action to take, using the policy
    #     """
    #     return np.random.choice(np.arange(self.act_dim), 1, p=self.actor.predict(s).ravel())[0]

    # def discount(self, r):
    #     """ Compute the gamma-discounted rewards over an episode
    #     """
    #     discounted_r, cumul_r = np.zeros_like(r), 0
    #     for t in reversed(range(0, len(r))):
    #         cumul_r = r[t] + cumul_r * self.gamma
    #         discounted_r[t] = cumul_r
    #     return discounted_r

    # def train_models(self, states, actions, rewards, done):
    #     """ Update actor and critic networks from experience
    #     """
    #     # Compute discounted rewards and Advantage (TD. Error)
    #     discounted_rewards = self.discount(rewards)
    #     state_values = self.critic.predict(np.array(states))
    #     advantages = discounted_rewards - np.reshape(state_values, len(state_values))
    #     # Networks optimization
    #     self.a_opt([states, actions, advantages])
    #     self.c_opt([states, discounted_rewards])

    # def train(self, env, args, summary_writer):
    #     """ Main A2C Training Algorithm
    #     """

    #     results = []
    #     global_rewards = []

    #     # Main Loop
    #     tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")
    #     for e in tqdm_e:

    #         # Reset episode
    #         time, cumul_reward, done = 0, 0, False
    #         old_state = env.reset()
    #         actions, states, rewards = [], [], []

    #         while not done:
    #             if args.render: env.render()
    #             # Actor picks an action (following the policy)
    #             a = self.policy_action(old_state)
    #             # Retrieve new state, reward, and whether the state is terminal
    #             new_state, r, done, _ = env.step(a)
    #             # Memorize (s, a, r) for training
    #             actions.append(to_categorical(a, self.act_dim))
    #             rewards.append(r)
    #             states.append(old_state)
    #             # Update current state
    #             old_state = new_state
    #             cumul_reward += r
    #             time += 1

    #         # Train using discounted rewards ie. compute updates
    #         self.train_models(states, actions, rewards, done)

    #         # Gather stats every episode for plotting
    #         if(args.gather_stats):
    #             mean, stdev = gather_stats(self, env)
    #             results.append([e, mean, stdev])
    #             global_rewards.append(cumul_reward)

    #         # Export results for Tensorboard
    #         score = tfSummary('score', cumul_reward)
    #         summary_writer.add_summary(score, global_step=e)
    #         summary_writer.flush()

    #         # Display score
    #         tqdm_e.set_description("Score: " + str(cumul_reward))
    #         tqdm_e.refresh()

    #     return global_rewards

    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)




    # ********************************************************
    # Colocar em outro lugar
    # ********************************************************

    def build_critic(self):
        HIDDEN_SIZE = 128
        NUM_LAYERS = 2
        state_input = Input(shape=(self.env_dim,))
        x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(HIDDEN_SIZE, activation='tanh')(x)

        out_value = Dense(1)(x)

        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=self.lr), loss='mse')
        model.summary()

        return model

    def build_actor(self):
        HIDDEN_SIZE = 128
        NUM_LAYERS = 2
        state_input = Input(shape=(self.env_dim,))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(self.act_dim,))

        x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(HIDDEN_SIZE, activation='tanh')(x)

        out_actions = Dense(self.act_dim, activation='softmax', name='output')(x)

        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=self.lr),
                      loss=[
                          proximal_policy_optimization_loss(
                            advantage=advantage,
                            old_prediction=old_prediction,
                            loss_clipping=self.loss_clipping,
                            entropy_loss=self.entropy_loss
                          )
                        ]
                    )
        model.summary()

        return model

    def transform_reward(self):
        GAMMA = 0.99
        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * GAMMA

    def get_action(self):
        DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, self.act_dim)), np.zeros((1, 1))

        p = self.actor.predict([self.observation.reshape(1, self.env_dim), DUMMY_VALUE, DUMMY_ACTION])
        if self.val is False:
            action = np.random.choice(self.act_dim, p=np.nan_to_num(p[0]))
        else:
            action = np.argmax(p[0])
        action_matrix = np.zeros(self.act_dim)
        action_matrix[action] = 1
        return action, action_matrix, p

    def get_batch(self, env):
        BUFFER_SIZE = 2048
        batch = [[], [], [], []]

        tmp_batch = [[], [], []]
        while len(batch[0]) < BUFFER_SIZE:
            env.render()

            action, action_matrix, predicted_action = self.get_action()
            observation, reward, done, info = env.step_one(action)

            self.reward.append(reward)

            tmp_batch[0].append(self.observation)
            tmp_batch[1].append(action_matrix)
            tmp_batch[2].append(predicted_action)
            self.observation = observation

            if done:
                self.transform_reward()
                if self.val is False:
                    for i in range(len(tmp_batch[0])):
                        obs, action, pred = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
                        r = self.reward[i]
                        batch[0].append(obs)
                        batch[1].append(action)
                        batch[2].append(pred)
                        batch[3].append(r)
                tmp_batch = [[], [], []]

                # Reset Env
                self.episode += 1
                if self.episode % 100 == 0:
                    self.val = True
                else:
                    self.val = False
                self.observation = env.reset_one()
                self.reward = []

        obs, action, pred, reward = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.reshape(np.array(batch[3]), (len(batch[3]), 1))
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        return obs, action, pred, reward

    def train(self, env, args, summary_writer):
        EPISODES = 100_000
        BATCH_SIZE = 4096
        BUFFER_SIZE = 2048
        EPOCHS=10

        self.observation = env.reset_one()
        self.reward = []
        self.reward_over_time = []
        self.gradient_steps = 0
        self.episode = 1

        while self.episode < EPISODES:
            print("Episode: ", self.episode)
            obs, action, pred, reward = self.get_batch(env)
            obs, action, pred, reward = obs[:BUFFER_SIZE], action[:BUFFER_SIZE], pred[:BUFFER_SIZE], reward[:BUFFER_SIZE]
            old_prediction = pred
            pred_values = self.critic.predict(obs)

            advantage = reward - pred_values

            actor_loss = self.actor.fit([obs, advantage, old_prediction], [action], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
            critic_loss = self.critic.fit([obs], [reward], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
            # summary_writer.add_scalar('Actor loss', actor_loss.history['loss'][-1], self.gradient_steps)
            # summary_writer.add_scalar('Critic loss', critic_loss.history['loss'][-1], self.gradient_steps)
            
            self.rewards.append(reward)
            self.actor_losses.append(actor_loss.history['loss'])
            self.critic_losses.append(critic_loss.history['loss'])

            self.gradient_steps += 1

        return rewards, actor_losses, critic_losses


def exponential_average(old, new, b1):
    return old * b1 + (1-b1) * new


def proximal_policy_optimization_loss(advantage, old_prediction, loss_clipping, entropy_loss):
    def loss(y_true, y_pred):
        prob = K.sum(y_true * y_pred, axis=-1)
        old_prob = K.sum(y_true * old_prediction, axis=-1)
        r = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - loss_clipping, max_value=1 + loss_clipping) * advantage) + entropy_loss * -(prob * K.log(prob + 1e-10)))
    return loss
# Advanced Deep RL in Keras

Modular Implementation of popular Deep Reinforcement Learning algorithms in Keras:

- [x] Synchronous N-step Advantage Actor Critic ([A2C](https://github.com/germain-hug/Advanced-Deep-RL-Keras#n-step-advantage-actor-critic-a2c))
- [x] Asynchronous N-step Advantage Actor-Critic ([A3C](https://github.com/germain-hug/Advanced-Deep-RL-Keras#n-step-asynchronous-advantage-actor-critic-a3c))
- [ ] Deep Deterministic Policy Gradient with Parameter Noise ([DDPG](https://github.com/germain-hug/Advanced-Deep-RL-Keras#deep-deterministic-policy-gradient-ddpg))
- [ ] Deep Deterministic Policy Gradient with Hindsight Experience Replay ([DDPG + HER](https://github.com/germain-hug/Advanced-Deep-RL-Keras#deep-deterministic-policy-gradient-with-hindsight-experience-replay-ddpg--her))
- [x] Double Deep Q-Network ([DDQN](https://github.com/germain-hug/Advanced-Deep-RL-Keras#double-deep-q-network-ddqn))
- [x] Double Deep Q-Network with Prioritized Experience Replay  ([DDQN + PER](https://github.com/germain-hug/Advanced-Deep-RL-Keras#deep-deterministic-policy-gradient-with-hindsight-experience-replay-ddpg--her))
- [ ] Dueling DDQN (DDQN)
- [ ] Rainbow
- [ ] REINFORCE
- [ ] DYNA-Q
- [ ] Proximal Policy Optimization (PPO)
- [ ] Impala
- [ ] Spiral

## Getting Started

This implementation requires keras 2.1.6, as well as OpenAI gym.
``` bash
$ pip install gym keras==2.1.6
```

### Arguments

| Argument | Description | Values |
| :---         |     :---      |          :--- |
| --type         |     Type of RL Algorithm to run      |  Choose from {A2C, A3C, DDQN, DDPG} |
| --env     | Specify the environemnt       | BreakoutNoFrameskip-v4 (default)      |
| --nb_episodes   | Number of episodes to run     | 5000 (default)    |
| --batch_size     | Batch Size (DDQN, DDPG)  | 32 (default)      |
| --consecutive_frames     | Number of stacked consecutive frames       | 4 (default)      |
| --is_atari     | Whether the environment is an Atari Game with pixel input   | -     |
| --with_PER     | Whether to use Prioritized Experience Replay (with DDQN)      | -      |
| --n_threads     | Number of threads (A3C)       | 16 (default)      |
| --gather_stats     | Whether to compute stats of scores averaged over 10 games (slow, see below)       | -      |
| --render     | Whether to render the environment as it is training       | -      |
| --gpu     | GPU index       | 0      |

<br />
<div align="center">
<img width="30%" src ="https://github.com/germain-hug/Advanced-Deep-RL-Keras/blob/master/results/a2c.gif?raw=true" />
<p style="text-align=center";> Results </p></div>  
<br />


# Actor-Critic Algorithms
### N-step Advantage Actor Critic (A2C)
The Actor-Critic algorithm is a model-free, off-policy method where the critic acts as a value-function approximator, and the actor as a policy-function approximator. When training, the critic predicts the TD-Error and guides the learning of both itself and the actor. In practice, we approximate the TD-Error using the Advantage function. For more stability, we use a shared computational backbone across both networks, as well as an N-step formulation of the discounted rewards. We also incorporate an entropy regularization term ("soft" learning) to encourage exploration. While A2C is simple and efficient, running it on Atari Games quickly becomes intractable due to long computation time.

### N-step Asynchronous Advantage Actor Critic (A3C)
In a similar fashion as the A2C algorithm, the implementation of A3C incorporates asynchronous weight updates, allowing for much faster computation. We use multiple agents to perform gradient ascent asynchronously, over multiple threads. We test A3C on the Atari Breakout environment.

### Deep Deterministic Policy Gradient (DDPG)
The DDPG algorithm is a model-free, off-policy algorithm for continuous action spaces. Similarly to A2C, it is an actor-critic algorithm in which the actor is trained on a deterministic target policy, and the critic predicts Q-Values. In order to reduce variance and increase stability, we use experience replay and separate target networks. Moreover, as hinted by [OpenAI](https://blog.openai.com/better-exploration-with-parameter-noise/), we encourage exploration through parameter space noise (as opposed to traditional action space noise). We test DDPG on the Lunar Lander environment.

### Deep Deterministic Policy Gradient with Hindsight Experience Replay (DDPG + HER)
Hindsight Experience Replay (HER) brings an improvement to both discrete and continuous action space off-policy methods. It is particularly suited for robotics application as it enables efficient learning from _sparse_ and _binary_ rewards. HER formulates the problem as a multi-goal task, where new goals are being sampled at the start of each episode through a specific strategy.

```bash
$ python3 main.py --type A2C --env CartPole-v1
$ python3 main.py --type A3C --env CartPole-v1 --nb_episodes 10000 --n_threads 16
$ python3 main.py --type A3C --env BreakoutNoFrameskip-v4 --is_atari --nb_episodes 10000 --n_threads 16
$ python3 main.py --type DDPG --env LunarLanderContinuous-v2
$ python3 main.py --type DDPG --env LunarLanderContinuous-v2 --with_HER
```

<br />
<div align="center">
<img width="60%" src ="https://github.com/germain-hug/Advanced-Deep-RL-Keras/blob/master/results/a2c.png?raw=true" /></div>  
<br />

# Deep Q-Learning Algorithms
### Double Deep Q-Network (DDQN)
The DQN algorithm is a Q-learning algorithm, which uses a Deep Neural Network as a Q-value function approximator. We estimate target Q-values by leveraging the Bellman equation, and gather experience through an epsilon-greedy policy. For more stability, we sample past experiences randomly (Experience Replay). A variant of the DQN algorithm is the Double-DQN (or DDQN). For a more accurate estimation of our Q-values, we use a second network to temper the overestimations of the Q-values by the original network. This _target_ network is updated at a slower rate Tau, at every training step.

### Double Deep Q-Network with Prioritized Experience Replay (DDQN + PER)
We can further improve our DDQN algorithm by adding in Prioritized Experience Replay (PER), which aims at performing importance sampling on the gathered experience. The experience is ranked by its TD-Error, and stored in a SumTree structure, which allows efficient retrieval of the _(s, a, r, s')_ transitions with the highest error.

```bash
$ python3 main.py --type DDQN --env CartPole-v1 --batch_size 64
$ python3 main.py --type DDQN --env CartPole-v1 --batch_size 64 --with_PER
```

<br />
<div align="center">
<img width="60%" src ="https://github.com/germain-hug/Advanced-Deep-RL-Keras/blob/master/results/ddqn.png?raw=true" /></div>  
<p> TODO Overlay DDQN / DDQN + PER <p/>
<br />

# Visualization & Monitoring

### Tensorboard monitoring
Using tensorboard, you can monitor the agent's score as it is training. When training, a log folder with the name matching the chosen environment will be created. For example, to follow the A2C progression on CartPole-v1, simply run:
```bash
$ tensorboard --logdir=A2C/tensorboard_CartPole-v1/
```
### Results plotting
When training with the argument --gather_stats, a log file is generated containing scores averaged over 10 games at every episode: `logs.csv`. Using [plotly](https://plot.ly/), you can visualize the average reward per episode.
To do so, you will first need to install plotly and get a [free licence](https://plot.ly/python/getting-started/).
```bash
pip3 install plotly
```
To set up your credentials, run:
```python
import plotly
plotly.tools.set_credentials_file(username='<your_username>', api_key='<your_key>')
```
Finally, to plot the results, run:
```bash
python3 utils/plot_results.py <path_to_your_log_file>
```

# Acknowledgments

- Atari Environment Helper Class template by [@ShanHaoYu](https://github.com/ShanHaoYu/Deep-Q-Network-Breakout/blob/master/environment.py)
- Atari Environment Wrappers by [OpenAI](github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py)
- SumTree Helper Class by [@jaara](https://github.com/jaara/AI-blog/blob/master/SumTree.py)

# References (Papers)

- [Advantage Actor Critic (A2C)](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf)
- [Asynchronous Advantage Actor Critic (A3C)](https://arxiv.org/pdf/1602.01783.pdf)
- [Deep Deterministic Policy Gradient (DDPG)](http://proceedings.mlr.press/v32/silver14.pdf)
- [Hindsight Experience Replay (HER)](https://arxiv.org/pdf/1707.01495.pdf)
- [Deep Q-Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- [Double Q-Learning](https://arxiv.org/pdf/1509.06461.pdf)
- [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)

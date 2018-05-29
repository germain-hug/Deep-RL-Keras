# Advanced Deep RL in Keras

Implementation of various Deep Reinforcement Learning algorithms:

- [x] N-step Advantage Actor Critic (A2C)
- [x] N-step Asynchronous Advantage Actor-Critic (A3C)
- [ ] Deep Deterministic Policy Gradient (DDPG)
- [ ] REINFORCE
- [ ] Deep Q-Learning (DQN)
- [ ] Dueling DQN (DDQN)
- [ ] Rainbow
- [ ] Proximal Policy Optimization (PPO)
- [ ] Impala
- [ ] Spiral

### Prerequisites

This implementation requires keras 2.1.6, as well as OpenAI gym.
```
pip install gym keras==2.1.6
```

## N-step Advantage Actor Critic (A2C)
The Actor-Critic algorithm is a Policy Gradient method where the critic acts as a value-function approximator, and the actor as a policy-function approximator. When training, the critic predicts the TD-Error and guides the learning of both itself and the actor. In practice, we approximate the TD-Error using the Advantage function. For more stability, we use a shared computational backbone across both networks, as well as an N-step formulation of the discounted rewards. We also incorporate an entropy regularization term ("soft" learning) to encourage exploration.  

```bash
python3 main.py --env CartPole-v1 --nb_episodes 10000 --render
```

Link to [[paper]](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf)

## N-step Asynchronous Advantage Actor Critic (A3C)
In a similar fashion as the A2C algorithm, we implement a variant incorporating asynchronous weight updates. To do so, we use multiple agents to perform gradient ascent, over multiple threads.

```bash
python3 main.py --env CartPole-v1 --nb_episodes 10000 --n_threads 16
```

Link to [[paper]](https://arxiv.org/pdf/1602.01783.pdf)

## Deep Deterministic Policy Gradient (DDPG)
The DDPG algorithm is a model-free, off-policy algorithm. Similarly to A2C, it is an actor-critic algorithm in which the actor is trained on a deterministic target policy, and the critic predicts Q-Values (using TD errors). In order to reduce variance and increase stability, we use experience replay and separate target networks.


# Acknowledgments

# References

- A2C : [[paper]](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf)
- A3C : [[paper]](https://arxiv.org/pdf/1602.01783.pdf)

# Advanced Deep RL in Keras

Implementation of various Deep Reinforcement Learning algorithms:

- [x] N-step Advantage Actor Critic (A2C)
- [x] N-step Asynchronous Advantage Actor-Critic (A3C)
- [ ] REINFORCE
- [ ] DQN
- [ ] DDQN
- [ ] Rainbow
- [ ] PPO
- [ ] Impala
- [ ] Spiral

### Prerequisites

This implementation requires keras 2.1.6, as well as OpenAI gym.
```
pip install gym keras==2.1.6
```

## N-step Advantage Actor Critic (A2C)
The Actor-Critic method is a Policy Gradient solving method where the gradients are scaled by a Value Function approximation, given by a critic. In this method, we update both the critic using the Bellman Equation, and the Actor using an Advantage factor. For more stability, we use a shared backbone, as well as an N-step formulation of the discounted rewards. We also incorporate an entropy regularization term ("soft" learning) to encourage exploration.  

```bash
python3 main.py --env CartPole-v1 --nb_episodes 10000 --render
```

## N-step Asynchronous Advantage Actor Critic (A3C)
In a similar fashion as the A2C algorithm, we implement a variant incorporating asynchronous weight updates. To do so, we use multiple agents to perform gradient ascent, over multiple threads.

```bash
python3 main.py --env CartPole-v1 --nb_episodes 10000 --n_threads 16
```

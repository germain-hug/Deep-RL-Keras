# Advanced Deep RL in Keras

- A2C : Good but unstable (entropy collapse because - Schulman et al.), Actor and Critic focus on one strategy together and not enough exploration -> Hence plot score peak then collapse. Need for an entropy regularization term ("soft" learning) in the loss function.
- A3C : Asynchronous (-> faster), RMSProp + Entropy regularization, Shared Convolutional Layers A <-> C, n-step forward formulation of discounted rewards
- DQN
- DDQN
- Rainbow
- PPO
- Impala
- Spiral

# Advanced Deep RL in Keras

- A2C : The Actor-Critic method is a Policy Gradient solving method where the gradients are scaled by a Value Function approximation, given by a critic. We therefore update both the Critic (using the Bellman Equation) and the Actor (using an "Advantage" factor). For more stability 

Good but unstable (entropy collapse because - Schulman et al.), Actor and Critic focus on one strategy together and not enough exploration -> Hence plot score peak then collapse. Need for an entropy regularization term ("soft" learning) in the loss function.
- A3C : Asynchronous (-> faster), RMSProp + Entropy regularization, Shared Convolutional Layers A <-> C, n-step forward formulation of discounted rewards
- REINFORCE :
- DQN
- DDQN
- Rainbow
- PPO
- Impala
- Spiral


# [Scalable Bayesian Inverse Reinforcement Learning](https://openreview.net/forum?id=4qR3coiNaIv)

### Alex J. Chan and Mihaela van der Schaar

### International Conference on Learning Representations (ICLR) 2021

Last Updated: 12 February 2021

Code Author: Alex Chan (ajc340@cam.ac.uk)

This repo contains a JAX based implementation of the Approximate Variational Reward Imitation Learning (AVRIL) algorithm. The code is ready to run on the control environments in the OpenAI Gym, with pre-run expert trajectories stored in the volume folder. 

Example usage:

```python
from model import avril
from utils import load_data

# First setup the data, I have provided a helper function for dealing 
# with the OpenAI gym control environemnts
inputs,targets,a_dim,s_dim = load_data('CartPole-v1',num_trajs=15)

# However AVRIL can handle any data appropriately formatted, that is inputs
# that are (state,next_state) pairs and targets that are (action, next_action)
# pairs:
# inputs = [num_pairs x 2 x state_dimension]
# targets = [num_pairs x 2 x action_dimension]

agent = avril(inputs,targets,s_dim,a_dim,state_only=False)

# You can define the reward to be state-only or state-action depending on use
# Train for set number of iterations with desired batch-size
agent.train(iters=5000,batch_size=32)

# Now test by rolling out in the live Gym environment

agent.gym_test('CartPole-v1')

```
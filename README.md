
# [Scalable Bayesian Inverse Reinforcement Learning](https://openreview.net/forum?id=4qR3coiNaIv)

### Alex J. Chan and Mihaela van der Schaar

### International Conference on Learning Representations (ICLR) 2021

 [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
 <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
 
Last Updated: 12 February 2021

Code Author: Alex J. Chan (ajc340@cam.ac.uk)

This repo contains a JAX based implementation of the Approximate Variational Reward Imitation Learning (AVRIL) algorithm. The code is ready to run on the control environments in the OpenAI Gym, with pre-run expert trajectories stored in the volume folder. 

This repo is pip installable - clone it, optionally create a virtual env, and install it (this will automatically install dependencies).

```shell
git clone git@github.com:XanderJC/scalable-birl.git

cd scalable-birl

pip install -e .
```

Example usage:

```python
from sbirl import avril, load_data

# First setup the data, I have provided a helper function for dealing 
# with the OpenAI gym control environemnts

inputs,targets,a_dim,s_dim = load_data('CartPole-v1',num_trajs=15)

# However, AVRIL can handle any data appropriately formatted, that is inputs
# that are (state,next_state) pairs and targets that are (action, next_action)
# pairs:
# inputs = [num_pairs x 2 x state_dimension]
# targets = [num_pairs x 2 x 1]

# You can define the reward to be state-only or state-action depending on use

agent = avril(inputs,targets,s_dim,a_dim,state_only=True)

# Train for set number of iterations with desired batch-size

agent.train(iters=5000,batch_size=64)

# Now test by rolling out in the live Gym environment

agent.gym_test('CartPole-v1')

```

This example can be run simply using:

```shell
python sbirl/models.py
```

### Citing 

If you use this software please cite as follows:

```
@inproceedings{chan2021scalable,
    title={Scalable {B}ayesian Inverse Reinforcement Learning},
    author={Alex James Chan and Mihaela van der Schaar},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=4qR3coiNaIv}
}
```

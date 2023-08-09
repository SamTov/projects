#!/usr/bin/env python
# coding: utf-8

# In[1]:


import flax.linen as nn
import numpy as np
import optax
import jax.numpy as jnp
import XXXX as srl
from XXXX.models.interaction_model import Action
from XXXX.observables import Observable
import jax
import h5py as hf
import matplotlib.pyplot as plt

import XXXX.engine.espresso as espresso
from XXXX.utils import utils

import pint


# ### Get System Runner and Helper Functions
# 
# After this, you should have a system-runner variable

# In[2]:


simulation_name = "find-center-0"
seed = np.random.randint(0, 3453276453)
temperature = TEMP
n_colloids = 10

ureg = pint.UnitRegistry()
md_params = espresso.MDParams(
    ureg=ureg,
    fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
    WCA_epsilon=ureg.Quantity(273.15, "kelvin") * ureg.boltzmann_constant,
    temperature=ureg.Quantity(temperature, "kelvin"),
    box_length=ureg.Quantity(1000, "micrometer"),
    time_slice=ureg.Quantity(1.0, "second"),  # model timestep
    time_step=ureg.Quantity(0.01, "second"),  # integrator timestep
    write_interval=ureg.Quantity(2, "second"),
)

system_runner = srl.espresso.EspressoMD(
    md_params=md_params,
    n_dims=2,
    seed=seed,
    out_folder='./training',
    write_chunk_size=100,
)

system_runner.add_colloids(
    n_colloids,
    ureg.Quantity(2.14, "micrometer"),
    ureg.Quantity(np.array([500, 500, 0]), "micrometer"),
    ureg.Quantity(100, "micrometer"),
    type_colloid=0,
)


# ### Set Parameters

# In[4]:


n_slices = int(ureg.Quantity(3, "minute") / md_params.time_slice)
n_episodes = 5000
episode_length = 20 #int(np.ceil(n_slices / 10))


# In[5]:


# Exploration policy
exploration_policy = srl.exploration_policies.RandomExploration(probability=0.2)

# Sampling strategy
sampling_strategy = srl.sampling_strategies.GumbelDistribution()

# Value function
value_function = srl.value_functions.ExpectedReturns(
    gamma=0.99, standardize=True
)

def scale_function(distance: float):
    """
    Scaling function for the task
    """
    return 1 / distance

# Set the task
task = srl.tasks.searching.GradientSensing(
    source=np.array([500.0, 500.0, 0.0]),
    decay_function=scale_function,
    reward_scale_factor=10,
    box_length=np.array([1000.0, 1000.0, 1000]),
)

observable = srl.observables.ConcentrationField(
    source=np.array([500.0, 500.0, 0.0]),
    decay_fn=scale_function,
    scale_factor=10,
    box_length=np.array([1000.0, 1000.0, 1000]),
)

# Define the loss model
loss = srl.losses.PolicyGradientLoss(value_function=value_function)

class ActorNet(nn.Module):
    """A simple dense model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=4)(x)
        return x

class CriticNet(nn.Module):
    """A simple dense model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x
    
actor = srl.networks.FlaxModel(
    flax_model=ActorNet(),
    optimizer=optax.adam(learning_rate=0.001),
    input_shape=(1,),
    sampling_strategy=sampling_strategy,
    exploration_policy=exploration_policy,
)

critic = srl.networks.FlaxModel(
    flax_model=CriticNet(),
    optimizer=optax.adam(learning_rate=0.001),
    input_shape=(1,),
)

translate = Action(force=10.0)
rotate_clockwise = Action(torque=np.array([0.0, 0.0, 10.0]))
rotate_counter_clockwise = Action(torque=np.array([0.0, 0.0, -10.0]))
do_nothing = Action()

actions = {
    "RotateClockwise": rotate_clockwise,
    "Translate": translate,
    "RotateCounterClockwise": rotate_counter_clockwise,
    "DoNothing": do_nothing,
}

protocol = srl.rl_protocols.ActorCritic(
    particle_type=0, 
    actor=actor, 
    critic=critic, 
    task=task, 
    observable=observable, 
    actions=actions
)


# In[9]:


rl_trainer = srl.gyms.Gym(
    [protocol],
    loss,
)


# In[10]:


rl_trainer.perform_rl_training(
    system_runner=system_runner,
    n_episodes=n_episodes,
    episode_length=episode_length,
)


# In[12]:


rl_trainer.export_models()


# ### Train without exploration

# In[ ]:


# Exploration policy
exploration_policy = srl.exploration_policies.RandomExploration(probability=0.0)

# Sampling strategy
sampling_strategy = srl.sampling_strategies.GumbelDistribution()

# Value function
value_function = srl.value_functions.ExpectedReturns(
    gamma=0.99, standardize=True
)

def scale_function(distance: float):
    """
    Scaling function for the task
    """
    return 1 / distance

# Set the task
task = srl.tasks.searching.GradientSensing(
    source=np.array([500.0, 500.0, 0.0]),
    decay_function=scale_function,
    reward_scale_factor=10,
    box_length=np.array([1000.0, 1000.0, 1000]),
)

observable = srl.observables.ConcentrationField(
    source=np.array([500.0, 500.0, 0.0]),
    decay_fn=scale_function,
    scale_factor=10,
    box_length=np.array([1000.0, 1000.0, 1000]),
)

# Define the loss model
loss = srl.losses.PolicyGradientLoss(value_function=value_function)

class ActorNet(nn.Module):
    """A simple dense model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=4)(x)
        return x

class CriticNet(nn.Module):
    """A simple dense model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x
    
actor = srl.networks.FlaxModel(
    flax_model=ActorNet(),
    optimizer=optax.adam(learning_rate=0.001),
    input_shape=(1,),
    sampling_strategy=sampling_strategy,
    exploration_policy=exploration_policy,
)

critic = srl.networks.FlaxModel(
    flax_model=CriticNet(),
    optimizer=optax.adam(learning_rate=0.001),
    input_shape=(1,),
)

translate = Action(force=10.0)
rotate_clockwise = Action(torque=np.array([0.0, 0.0, 10.0]))
rotate_counter_clockwise = Action(torque=np.array([0.0, 0.0, -10.0]))
do_nothing = Action()

actions = {
    "RotateClockwise": rotate_clockwise,
    "Translate": translate,
    "RotateCounterClockwise": rotate_counter_clockwise,
    "DoNothing": do_nothing,
}

protocol = srl.rl_protocols.ActorCritic(
    particle_type=0, 
    actor=actor, 
    critic=critic, 
    task=task, 
    observable=observable, 
    actions=actions
)


# In[13]:


rl_trainer = srl.gyms.Gym(
    [protocol],
    loss
)


# In[14]:


rl_trainer.restore_models()


# In[17]:


rl_trainer.perform_rl_training(
    system_runner=system_runner,
    n_episodes=n_episodes,
    episode_length=episode_length,
)
rl_trainer.export_models()


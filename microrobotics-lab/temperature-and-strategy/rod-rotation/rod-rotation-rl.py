#!/usr/bin/env python
# coding: utf-8

# In[2]:


import flax.linen as nn
import numpy as np
import optax
import jax.numpy as jnp
import swarmrl as srl
from swarmrl.models.interaction_model import Action
from swarmrl.observables import Observable
import jax
import h5py as hf
import matplotlib.pyplot as plt

import swarmrl.engine.espresso as espresso
from swarmrl.utils import utils

import pint


# ### Get System Runner and Helper Functions
# 
# After this, you should have a system-runner variable

# In[4]:


simulation_name = "rod-rotation"
seed = np.random.randint(0, 3453276453)

temperature = TEMP
n_colloids = 20

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
system_runner.add_rod(
    rod_center=ureg.Quantity([500, 500, 0], "micrometer"),
    rod_length=ureg.Quantity(100, "micrometer"),
    rod_thickness=ureg.Quantity(100 / 59, "micrometer"),
    rod_start_angle=90.0,
    n_particles=59,
    friction_trans=ureg.Quantity(4.388999e-7, "newton * second / meter"),
    friction_rot=ureg.Quantity(6.902407326e-16, "newton * second * meter"),
    rod_particle_type=1,
)


# ## Set up the RL Parameters

# In[5]:


n_slices = int(ureg.Quantity(3, "minute") / md_params.time_slice)
n_episodes = 5000
episode_length = 20 #    int(np.ceil(n_slices / 10))


# In[6]:


# Exploration policy
exploration_policy = srl.exploration_policies.RandomExploration(probability=0.0)

# Sampling strategy
sampling_strategy = srl.sampling_strategies.GumbelDistribution()

# Value function
value_function = srl.value_functions.ExpectedReturns(
    gamma=0.99, standardize=True
)

def decay_fn(r: float):
    return 1 / r

observable = srl.observables.SubdividedVisionCones(
    vision_range=1000000.0,
    vision_half_angle=1.4,
    n_cones=5,
    radii=jnp.array([3.08 for _ in range(n_colloids)] + [0.84745763 for _ in range(59)])
)

# Define the loss model
loss = srl.losses.PolicyGradientLoss(value_function=value_function)

rotation_task = srl.tasks.object_movement.RotateRod(
    particle_type=0,
    angular_velocity_scale=1000,
    rod_type=1,
    direction="CCW",
    partition=False
)

rod_finding_task = srl.tasks.searching.SpeciesSearch(
    particle_type=0,
    decay_fn=decay_fn,
    box_length=np.array([1000.0, 1000.0, 1000.0]),
    scale_factor=100,
    sensing_type=1
)

total_task = srl.tasks.MultiTasking(particle_type=0, tasks=[rotation_task, rod_finding_task])

class ColloidEmbedding(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=2)(x)
    
class RodEmbedding(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=2)(x)
    
class ActorNet(nn.Module):
    """A simple dense model."""
    
    def setup(self):
        self.colloid_embedding = ColloidEmbedding()
        self.rod_embeding = RodEmbedding()
    
    @nn.compact
    def __call__(self, x):
        colloid_embedding = self.colloid_embedding(x[:, 0])
        rod_embedding = self.rod_embeding(x[:, 1])
        
        x = colloid_embedding + rod_embedding
        x = nn.relu(x)
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=4)(x)
        return x
    
class CriticNet(nn.Module):
    """A simple dense model."""
    
    def setup(self):
        self.colloid_embedding = ColloidEmbedding()
        self.rod_embeding = RodEmbedding()
    
    @nn.compact
    def __call__(self, x):
        colloid_embedding = self.colloid_embedding(x[:, 0])
        rod_embedding = self.rod_embeding(x[:, 1])
        
        x = colloid_embedding + rod_embedding
        x = nn.relu(x)
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x
    
actor = srl.networks.FlaxModel(
    flax_model=ActorNet(),
    optimizer=optax.adam(learning_rate=0.001),
    input_shape=(1, 5, 2),
    sampling_strategy=sampling_strategy,
    exploration_policy=exploration_policy,
)

critic = srl.networks.FlaxModel(
    flax_model=CriticNet(),
    optimizer=optax.adam(learning_rate=0.001),
    input_shape=(1, 5, 2),
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
    task=rod_finding_task, 
    observable=observable, 
    actions=actions
)
rl_trainer = srl.gyms.Gym(
    [protocol],
    loss,
)
rl_trainer.perform_rl_training(
    system_runner=system_runner,
    n_episodes=n_episodes,
    episode_length=episode_length,
)
rl_trainer.export_models()

# Exploration policy
exploration_policy = srl.exploration_policies.RandomExploration(probability=0.0)

# Sampling strategy
sampling_strategy = srl.sampling_strategies.GumbelDistribution()

# Value function
value_function = srl.value_functions.ExpectedReturns(
    gamma=0.99, standardize=True
)

observable = srl.observables.SubdividedVisionCones(
    vision_range=1000000.0,
    vision_half_angle=1.4,
    n_cones=5,
    radii=jnp.array([3.08 for _ in range(n_colloids)] + [0.84745763 for _ in range(59)])
)

# Define the loss model
loss = srl.losses.PolicyGradientLoss(value_function=value_function)    
    
actor = srl.networks.FlaxModel(
    flax_model=ActorNet(),
    optimizer=optax.adam(learning_rate=0.001),
    input_shape=(1, 5, 2),
    sampling_strategy=sampling_strategy,
    exploration_policy=exploration_policy,
)

critic = srl.networks.FlaxModel(
    flax_model=CriticNet(),
    optimizer=optax.adam(learning_rate=0.001),
    input_shape=(1, 5, 2),
)

protocol = srl.rl_protocols.ActorCritic(
    particle_type=0, 
    actor=actor, 
    critic=critic, 
    task=rotation_task, 
    observable=observable, 
    actions=actions
)
rl_trainer = srl.gyms.Gym(
    [protocol],
    loss,
)
rl_trainer.restore_models()
rl_trainer.perform_rl_training(
    system_runner=system_runner,
    n_episodes=n_episodes,
    episode_length=episode_length,
)
rl_trainer.export_models()


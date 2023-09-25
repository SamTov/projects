#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import flax.linen as nn
import numpy as np
import optax

import swarmrl as srl
from swarmrl.tasks import Task
from swarmrl.observables import Observable
from swarmrl.models.interaction_model import Action

import swarmrl.engine.espresso as espresso
from swarmrl.utils import utils

import pint
import h5py as hf
import matplotlib.pyplot as plt


# # Experiment Parameters

# In[2]:


prey_knowledge = 2
predator_knowledge = KNOWLEDGE


# # Simulation Definition

# In[3]:


simulation_name = "predator-prey"
seed = 42
temperature = 300

prey = 100
predators = 5

ureg = pint.UnitRegistry()
md_params = espresso.MDParams(
    ureg=ureg,
    fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
    WCA_epsilon=ureg.Quantity(temperature, "kelvin") * ureg.boltzmann_constant,
    temperature=ureg.Quantity(temperature, "kelvin"),
    box_length=ureg.Quantity(1000, "micrometer"),
    time_slice=ureg.Quantity(0.5, "second"),  # model timestep
    time_step=ureg.Quantity(0.01, "second"),  # integrator timestep
    write_interval=ureg.Quantity(2, "second"),
)

system_runner = srl.espresso.EspressoMD(
    md_params=md_params,
    n_dims=2,
    seed=42,
    out_folder='./training',
    write_chunk_size=100,
)

coll_type = 0
system_runner.add_colloids(
    prey,
    ureg.Quantity(4.0, "micrometer"),
    ureg.Quantity(np.array([500, 500, 0]), "micrometer"),
    ureg.Quantity(100, "micrometer"),
    type_colloid=0,
)

coll_type = 1
system_runner.add_colloids(
    predators,
    ureg.Quantity(2.00, "micrometer"),
    ureg.Quantity(np.array([500, 500, 0]), "micrometer"),
    ureg.Quantity(800, "micrometer"),
    type_colloid=1,
)


# # RL Training Parameters

# In[4]:


n_episodes = 10000
episode_length = 30

loss = srl.losses.PolicyGradientLoss(
    value_function=srl.value_functions.ExpectedReturns(
        gamma=0.99, standardize=True
    )
)


# ## Decay Function

# In[5]:


def decay_fn(r: float):
    return 1 / r


# ## Networks

# In[6]:


class PreyEmbedding(nn.Module):
    knowledge : int
        
    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=self.knowledge)(x)
    
class PredatorEmbedding(nn.Module):
    knowledge : int
        
    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=self.knowledge)(x)
    
class ActorNet(nn.Module):
    """A simple dense model."""
    knowledge : int
        
    def setup(self):
        self.colloid_embedding = PreyEmbedding(
            knowledge=self.knowledge
        )
        self.rod_embeding = PredatorEmbedding(
            knowledge=self.knowledge
        )
    
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
    knowledge : int
        
    def setup(self):
        self.colloid_embedding = PreyEmbedding(
            knowledge=self.knowledge
        )
        self.rod_embeding = PredatorEmbedding(
            knowledge=self.knowledge
        )
    
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


# # Prey Model Setup

# ## Prey Task

# In[7]:


# Task for avoiding the predator.
prey_avoidance_task = srl.tasks.searching.SpeciesSearch(
    decay_fn=decay_fn,
    box_length=np.array([1000.0, 1000.0, 1000.0]),
    sensing_type=1,  # Sense and avoid type 1
    particle_type=0,
    scale_factor=10,
    avoid=True
)

# Community reward
prey_community_task = srl.tasks.searching.SpeciesSearch(
    decay_fn=decay_fn,
    box_length=np.array([1000.0, 1000.0, 1000.0]),
    sensing_type=0,  # Sense and avoid type 1
    particle_type=0,
    scale_factor=5,
    avoid=False
)

# Final prey reward is a linear combination.
prey_task = srl.tasks.MultiTasking(
    particle_type=0, 
    tasks=[prey_avoidance_task, prey_community_task]
)


# ## Prey Observable

# In[8]:


prey_observable = srl.observables.SubdividedVisionCones(
    vision_range=1000000.0,
    vision_half_angle=1.4,
    n_cones=5,
    radii=np.array([4.0 for _ in range(prey)] + [2.0 for _ in range(predators)])
)


# ## Prey Action Space

# In[9]:


prey_translate = Action(force=10.0)
prey_cw = Action(torque=np.array([0.0, 0.0, 10.0]))
prey_ccw = Action(torque=np.array([0.0, 0.0, -10.0]))
prey_dn = Action()

prey_actions = {
    "RotateClockwise": prey_cw,
    "Translate": prey_translate,
    "RotateCounterClockwise": prey_ccw,
    "DoNothing": prey_dn,
}


# ## Prey Protocol

# In[10]:


prey_actor = srl.networks.FlaxModel(
    flax_model=ActorNet(prey_knowledge),
    optimizer=optax.adam(learning_rate=0.002),
    input_shape=(1, 5, 2),
    sampling_strategy=srl.sampling_strategies.GumbelDistribution(),
    exploration_policy=srl.exploration_policies.RandomExploration(
        probability=0.0
    ),
)

prey_critic = srl.networks.FlaxModel(
    flax_model=CriticNet(prey_knowledge),
    optimizer=optax.adam(learning_rate=0.002),
    input_shape=(1, 5, 2),
)

prey_protocol = srl.rl_protocols.ActorCritic(
    particle_type=0, 
    actor=prey_actor, 
    critic=prey_critic, 
    task=prey_task,
    observable=prey_observable, 
    actions=prey_actions
)


# # Predator Model Setup

# ## Predator Task

# In[11]:


predator_task = srl.tasks.searching.SpeciesSearch(
    decay_fn=decay_fn,
    box_length=np.array([1000.0, 1000.0, 1000.0]),
    sensing_type=0,  # Sense and catch type 0
    particle_type=1,
    scale_factor=50,
    avoid=False
)


# ## Predator Observable

# In[12]:


predator_observable = srl.observables.SubdividedVisionCones(
    vision_range=1000000.0,
    vision_half_angle=1.4,
    particle_type=1,
    n_cones=5,
    radii=np.array([4.0 for _ in range(prey)] + [2.0 for _ in range(predators)])
)


# ## Predator Action Space

# In[13]:


predator_translate = Action(force=20.0)
predator_cw = Action(torque=np.array([0.0, 0.0, 10.0]))
predator_ccw = Action(torque=np.array([0.0, 0.0, -10.0]))
predator_dn = Action()

predator_actions = {
    "RotateClockwise": predator_cw,
    "Translate": predator_translate,
    "RotateCounterClockwise": predator_ccw,
    "DoNothing": predator_dn,
}


# ## Predator Protocol

# In[14]:


predator_actor = srl.networks.FlaxModel(
    flax_model=ActorNet(predator_knowledge),
    optimizer=optax.adam(learning_rate=0.002),
    input_shape=(1, 5, 2),
    sampling_strategy=srl.sampling_strategies.GumbelDistribution(),
    exploration_policy=srl.exploration_policies.RandomExploration(
        probability=0.0
    ),
)

predator_critic = srl.networks.FlaxModel(
    flax_model=CriticNet(predator_knowledge),
    optimizer=optax.adam(learning_rate=0.002),
    input_shape=(1, 5, 2),
)

predator_protocol = srl.rl_protocols.ActorCritic(
    particle_type=1, 
    actor=predator_actor, 
    critic=predator_critic, 
    task=predator_task, 
    observable=predator_observable,
    actions=predator_actions
)


# # RL Training

# In[15]:


rl_trainer = srl.gyms.Gym(
    [prey_protocol, predator_protocol],
    loss,
)


# In[ ]:


rl_trainer.perform_rl_training(
    system_runner=system_runner,
    n_episodes=n_episodes,
    episode_length=episode_length,
)
rl_trainer.export_models()


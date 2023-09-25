#!/usr/bin/env python
# coding: utf-8

# In[2]:


# SwarmRL Imports
import swarmrl as srl
import swarmrl.engine.espresso as espresso
from swarmrl.utils import utils
from swarmrl.models.interaction_model import Action
from swarmrl.observables import Observable

# Linalg imports
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import optax
import jax

# Helper imports
import pint
import h5py as hf
import matplotlib.pyplot as plt


# In[3]:


simulation_name = "rod-rotation-embedding-study"
seed = 42

temperature = 150.0
n_colloids = 20

ureg = pint.UnitRegistry()

md_params = espresso.MDParams(
    ureg=ureg,
    fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
    WCA_epsilon=ureg.Quantity(temperature, "kelvin") * ureg.boltzmann_constant,
    temperature=ureg.Quantity(temperature, "kelvin"),
    box_length=ureg.Quantity(1000, "micrometer"),
    time_slice=ureg.Quantity(0.2, "second"),  # model timestep
    time_step=ureg.Quantity(0.004, "second"),  # integrator timestep
    write_interval=ureg.Quantity(2, "second"),
)

system_runner = srl.espresso.EspressoMD(
    md_params=md_params,
    n_dims=2,
    seed=seed,
    out_folder='./deployment',
    write_chunk_size=100,
)

system_runner.add_colloids(
    n_colloids,
    ureg.Quantity(3.08, "micrometer"),
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


# In[4]:


class ColloidEmbedding(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=KNOWLEDGE)(x)
    
class RodEmbedding(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=KNOWLEDGE)(x)
    
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


# In[5]:


simulation_length=50000

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
observable.initialize(system_runner.colloids)
# Define the loss model

rotation_task = srl.tasks.object_movement.RotateRod(
    particle_type=0,
    angular_velocity_scale=100,
    rod_type=1,
    direction="CCW",
    partition=True
)

rod_finding_task = srl.tasks.searching.SpeciesSearch(
    particle_type=0,
    decay_fn=decay_fn,
    box_length=np.array([1000.0, 1000.0, 1000.0]),
    scale_factor=10,
    sensing_type=1
)

total_task = srl.tasks.MultiTasking(particle_type=0, tasks=[rotation_task, rod_finding_task])
total_task.initialize(system_runner.colloids)

actor = srl.networks.FlaxModel(
    flax_model=ActorNet(),
    optimizer=optax.adam(learning_rate=0.001),
    input_shape=(1, 5, 2),
    sampling_strategy=sampling_strategy,
    exploration_policy=exploration_policy,
)
actor.restore_model_state(filename="ActorModel_0", directory="Models/")

force_fn = srl.models.ml_model.MLModel(
    models={"0": actor},
    observables={"0": observable},
    tasks={"0": total_task},
    actions={"0": actions},
    record_traj=False  # Only used during training, turn it off here.
)

system_runner.integrate(simulation_length, force_model=force_fn)


# In[ ]:





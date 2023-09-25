#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# # Simulation Setup

# In[ ]:


simulation_name = "rod-rotation"
seed = np.random.randint(0, 3453276453)

temperature = TEMP
n_colloids = 30

offset = 0.
if temperature == 0.:
    offset = 5.

ureg = pint.UnitRegistry()
md_params = espresso.MDParams(
    ureg=ureg,
    fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
    WCA_epsilon=ureg.Quantity(temperature + offset, "kelvin") * ureg.boltzmann_constant,
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
    out_folder='./deployment_large',
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


# In[ ]:


simulation_length = 50000


# # Restore Models

# In[ ]:


exploration_policy = srl.exploration_policies.RandomExploration(probability=0.0)

# Sampling strategy
sampling_strategy = srl.sampling_strategies.GumbelDistribution()


def decay_fn(r: float):
    return 1 / r

observable = srl.observables.SubdividedVisionCones(
    vision_range=1000000.0,
    vision_half_angle=1.4,
    n_cones=5,
    radii=jnp.array([3.08 for _ in range(n_colloids)] + [0.84745763 for _ in range(59)])
)


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
    scale_factor=10,
    sensing_type=1
)

total_task = srl.tasks.MultiTasking(particle_type=0, tasks=[rotation_task, rod_finding_task])

observable.initialize(system_runner.colloids)
total_task.initialize(system_runner.colloids)

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
    
actor = srl.networks.FlaxModel(
    flax_model=ActorNet(),
    optimizer=optax.adam(learning_rate=0.001),
    input_shape=(1, 5, 2),
    sampling_strategy=sampling_strategy,
    exploration_policy=exploration_policy,
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
actor.restore_model_state(filename="ActorModel_0", directory="Models/")


# # Run the simulation

# In[ ]:


force_fn = srl.models.ml_model.MLModel(
    models={"0": actor},
    observables={"0": observable},
    tasks={"0": total_task},
    actions={"0": actions},
    record_traj=False  # Only used during training, turn it off here.
)


# In[ ]:


system_runner.integrate(simulation_length, force_model=force_fn)


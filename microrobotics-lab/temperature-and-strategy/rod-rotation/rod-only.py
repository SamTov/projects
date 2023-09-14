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

ureg = pint.UnitRegistry()
md_params = espresso.MDParams(
    ureg=ureg,
    fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
    WCA_epsilon=ureg.Quantity(273.15, "kelvin") * ureg.boltzmann_constant,
    temperature=ureg.Quantity(temperature, "kelvin"),
    box_length=ureg.Quantity(1000, "micrometer"),
    time_slice=ureg.Quantity(1.0, "second"),  # model timestep
    time_step=ureg.Quantity(0.01, "second"),  # integrator timestep
    write_interval=ureg.Quantity(0.1, "second"),
)

system_runner = srl.espresso.EspressoMD(
    md_params=md_params,
    n_dims=2,
    seed=seed,
    out_folder='./no_colloids',
    write_chunk_size=100,
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


system_runner.integrate(simulation_length)


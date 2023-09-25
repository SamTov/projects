# Imports
import flax.linen as nn
import numpy as np
import optax
import jax.numpy as jnp
import swarmrl as srl
from swarmrl.models.interaction_model import Action
from swarmrl.observables import Observable
import matplotlib.pyplot as plt

import swarmrl.engine.espresso as espresso
from swarmrl.utils import utils

import pint


# Define global properties.
simulation_name = "chemotaxis"
seed = np.random.randint(0, 3453276453)
temperature = TEMPERATURE
radius = RADIUS
n_colloids = 10

# Set up the MD parameters.
ureg = pint.UnitRegistry()
md_params = espresso.MDParams(
    ureg=ureg,
    fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
    WCA_epsilon=ureg.Quantity(temperature, "kelvin") * ureg.boltzmann_constant,
    temperature=ureg.Quantity(temperature, "kelvin"),
    box_length=ureg.Quantity(1000, "micrometer"),
    time_slice=ureg.Quantity(0.1, "second"),  # model timestep
    time_step=ureg.Quantity(0.0001, "second"),  # integrator timestep
    write_interval=ureg.Quantity(1, "second"),
)

system_runner = srl.espresso.EspressoMD(
    md_params=md_params,
    n_dims=2,
    seed=seed,
    out_folder='./training',
    write_chunk_size=100,
)

# Add the particles
aspect_ratio = 3. # Prolate particle

# calculate semiaxes via volume equivalent
equatorial_semiaxis = ureg.Quantity(radius, "micrometer") / np.cbrt(aspect_ratio)
axial_semiaxis = equatorial_semiaxis * aspect_ratio

gamma_trans_ax, gamma_trans_eq = (
    srl.utils.calc_ellipsoid_friction_factors_translation(
        axial_semiaxis, equatorial_semiaxis, md_params.fluid_dyn_viscosity
    )
)
gamma_rot_ax, gamma_rot_eq = (
    srl.utils.calc_ellipsoid_friction_factors_rotation(
        axial_semiaxis, equatorial_semiaxis, md_params.fluid_dyn_viscosity
    )
)

gamma_trans = srl.utils.convert_array_of_pint_to_pint_of_array(
    [gamma_trans_eq, gamma_trans_eq, gamma_trans_ax], ureg
)
gamma_rot = srl.utils.convert_array_of_pint_to_pint_of_array(
    [gamma_rot_eq, gamma_rot_eq, gamma_rot_ax], ureg
)


system_runner.add_colloids(
    n_colloids,
    equatorial_semiaxis,
    ureg.Quantity(np.array([500, 500, 0]), "micrometer"),
    ureg.Quantity(100, "micrometer"),
    type_colloid=0,
    gamma_translation=gamma_trans,
    gamma_rotation=gamma_rot,
    aspect_ratio=aspect_ratio
)


# Compute action space

swim_speed = ureg.Quantity(SPEED * 2  * radius, "micrometer / second")
angular_velocity = ureg.Quantity(600 * np.pi / 180, "1/second")

gamma, gamma_rotation = gamma_trans_ax, gamma_rot_eq

act_force = swim_speed * gamma
act_torque = angular_velocity* gamma_rotation


translate = Action(force=act_force.m_as("sim_force"))
rotate_clockwise = Action(torque=np.array([0.0, 0.0, act_torque.m_as("sim_torque")]))
rotate_counter_clockwise = Action(torque=np.array([0.0, 0.0, -1 * act_torque.m_as("sim_torque")]))
do_nothing = Action()

actions = {
    "RotateClockwise": rotate_clockwise,
    "Translate": translate,
    "RotateCounterClockwise": rotate_counter_clockwise,
    "DoNothing": do_nothing,
}

# Set Simulation running parameters
n_episodes = 5000
episode_length = 20

# #################### #
# Run with exploration #
# #################### #

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

protocol = srl.rl_protocols.ActorCritic(
    particle_type=0, 
    actor=actor, 
    critic=critic, 
    task=task, 
    observable=observable, 
    actions=actions
)


rl_trainer = srl.gyms.Gym(
    [protocol],
    loss,
)

rewards = rl_trainer.perform_rl_training(
    system_runner=system_runner,
    n_episodes=n_episodes,
    episode_length=episode_length,
)

rl_trainer.export_models()

np.save("rewards_exploration.npy", rewards)


# Train without exploration

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

protocol = srl.rl_protocols.ActorCritic(
    particle_type=0, 
    actor=actor, 
    critic=critic, 
    task=task, 
    observable=observable, 
    actions=actions
)


rl_trainer = srl.gyms.Gym(
    [protocol],
    loss
)

rl_trainer.restore_models()


rewards = rl_trainer.perform_rl_training(
    system_runner=system_runner,
    n_episodes=n_episodes,
    episode_length=episode_length,
)
rl_trainer.export_models()

np.save("rewards_no_exploration.npy", rewards)


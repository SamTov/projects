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
simulation_name = "chemotaxis-deploy"
seed = np.random.randint(0, 3453276453)
temperature = 300
radius = 2.425
n_colloids = 5

# Set up the MD parameters.
ureg = pint.UnitRegistry()
md_params = espresso.MDParams(
    ureg=ureg,
    fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
    WCA_epsilon=ureg.Quantity(temperature, "kelvin") * ureg.boltzmann_constant,
    temperature=ureg.Quantity(temperature, "kelvin"),
    box_length=ureg.Quantity(1000, "micrometer"),
    time_slice=ureg.Quantity(0.1, "second"),  # model timestep
    time_step=ureg.Quantity(0.0005, "second"),  # integrator timestep
    write_interval=ureg.Quantity(1, "second"),
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
    ureg.Quantity(radius, "micrometer"),
    ureg.Quantity(np.array([500, 500, 0]), "micrometer"),
    ureg.Quantity(100, "micrometer"),
    type_colloid=0,
)

# Compute actions space

# 2  diameters per second (5 * 2 * radius)
swim_speed = ureg.Quantity(2 * 2  * radius, "micrometer / second")
angular_velocity = ureg.Quantity(600 * np.pi / 180, "1/second")

gamma, gamma_rotation = system_runner.get_friction_coefficients(0)  # get for colloid 0

act_force = swim_speed.m_as("sim_velocity") * gamma
act_torque = angular_velocity.m_as("1 / sim_time") * gamma_rotation

translate = Action(force=act_force)
rotate_clockwise = Action(torque=np.array([0.0, 0.0, act_torque]))
rotate_counter_clockwise = Action(torque=np.array([0.0, 0.0, -1 * act_torque]))
do_nothing = Action()

actions = {
    "RotateClockwise": rotate_clockwise,
    "Translate": translate,
    "RotateCounterClockwise": rotate_counter_clockwise,
    "DoNothing": do_nothing,
}

# Set Simulation running parameters
simulation_length = 10000

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

observable.initialize(system_runner.colloids)
task.initialize(system_runner.colloids)

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

# Load the model state
actor.restore_model_state(filename="ActorModel_0", directory="Models/")

force_fn = srl.models.ml_model.MLModel(
    models={"0": actor},
    observables={"0": observable},
    tasks={"0": task},
    actions={"0": actions},
    record_traj=False  # Only used during training, turn it off here.
)

system_runner.integrate(simulation_length, force_model=force_fn)


# Oryx runs with GPU 1
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

# SwarmRL Imports
import swarmrl as srl
import swarmrl.engine.espresso as espresso
from swarmrl.actions import Action
from swarmrl.components import Colloid

# espresso imports
import espressomd

# Linalg imports
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import optax

# Helper imports
import pint
from typing import List   


system = espressomd.System(box_l=[1, 2, 3])
def get_engine(system_runner):
    # Simulation parameters
    seed = np.random.randint(645153513)

    temperature = TEMP
    n_colloids = 20

    ureg = pint.UnitRegistry()

    md_params = espresso.MDParams(
        ureg=ureg,
        fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
        WCA_epsilon=ureg.Quantity(1.0 + temperature, "kelvin") * ureg.boltzmann_constant,
        temperature=ureg.Quantity(temperature, "kelvin"),
        box_length=ureg.Quantity(150, "micrometer"),
        time_slice=ureg.Quantity(10.0, "second"),  # model timestep
        time_step=ureg.Quantity(0.001, "second"),  # integrator timestep
        write_interval=ureg.Quantity(1, "second"),
    )

    system_runner = srl.espresso.EspressoMD(
        md_params=md_params,
        n_dims=2,
        seed=seed,
        out_folder=f'deployment',
        write_chunk_size=100,
        system=system_runner,
        periodic=False
    )

    system_runner.add_colloids(
        n_colloids,
        ureg.Quantity(3.08, "micrometer"),
        ureg.Quantity(np.array([75., 75., 0]), "micrometer"),
        ureg.Quantity(25, "micrometer"),
        type_colloid=0,
    )
    system_runner.add_rod(
        rod_center=ureg.Quantity([75., 75., 0], "micrometer"),
        rod_length=ureg.Quantity(100, "micrometer"),
        rod_thickness=ureg.Quantity(100 / 59, "micrometer"),
        rod_start_angle=90.0,
        n_particles=59,
        friction_trans=ureg.Quantity(4.388999e-7, "newton * second / meter"),
        friction_rot=ureg.Quantity(6.902407326e-16, "newton * second * meter"),
        rod_particle_type=1,
    )

    # system_runner.add_confining_walls(wall_type=2)

    return system_runner

    
class ActorCriticNetwork(nn.Module):
    """A simple AC network."""
    
    def setup(self):
        # self.colloid_embedding = ColloidEmbedding()
        # self.rod_embeding = RodEmbedding()

        self.kernel_init = nn.initializers.xavier_normal()
    
    @nn.compact
    def __call__(self, x):
        vision_embedding = jnp.concatenate([x[:, 0], x[:, 1]], axis=-1)

        # Actor pass
        x = nn.Dense(
            features=128, 
            kernel_init=self.kernel_init, 
            )(vision_embedding)
        x = nn.leaky_relu(x)
        x = nn.Dense(
            features=128, 
            kernel_init=self.kernel_init, 
            )(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(
            features=128, 
            kernel_init=self.kernel_init, 
            )(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(
            features=128, 
            kernel_init=self.kernel_init, 
            )(x)
        x = nn.leaky_relu(x)

        # Critic pass
        y = nn.Dense(
            features=128, 
            kernel_init=self.kernel_init, 
            )(vision_embedding)
        y = nn.leaky_relu(y)
        y = nn.Dense(
            features=128, 
            kernel_init=self.kernel_init, 
            )(y)
        y = nn.leaky_relu(y)
        y = nn.Dense(
            features=128, 
            kernel_init=self.kernel_init, 
            )(y)
        y = nn.leaky_relu(y)
        y = nn.Dense(
            features=128, 
            kernel_init=self.kernel_init, 
            )(y)
        y = nn.leaky_relu(y)

        # Actor head
        actions = nn.Dense(
            features=4, 
            kernel_init=self.kernel_init, 
            )(x)

        # Critic head
        value = nn.Dense(
            features=1, 
            kernel_init=self.kernel_init, 
            )(y)

        return actions, value
    

# Exploration policy
exploration_policy = srl.exploration_policies.RandomExploration(probability=0.0)

# Sampling strategy
sampling_strategy = srl.sampling_strategies.GumbelDistribution()

# Loss function
value_function = srl.value_functions.GAE(gamma=0.995, lambda_=0.97)
loss = srl.losses.ProximalPolicyLoss(
    entropy_coefficient=0.01, epsilon=0.03
    )

observable = srl.observables.SubdividedVisionCones(
    vision_range=1000000.0,
    vision_half_angle=1.4,
    n_cones=5,
    radii=jnp.array([3.08 for _ in range(20)] + [0.84745763 for _ in range(59)])
)

rotation_task = srl.tasks.object_movement.RotateRod(
    particle_type=0,
    angular_velocity_scale=100000,
    rod_type=1,
    direction="CCW",
    partition=True,
    velocity_history=20
)

def decay_fn(x):
    return 1.0 - x

search_task = srl.tasks.searching.SpeciesSearch(
        decay_fn=decay_fn,
        box_length = np.array([150.0, 150.0, 150.0]),
        sensing_type = 1,
        avoid = False,
        scale_factor = 1,
        particle_type = 0,
)

find_and_rotate = srl.tasks.MultiTasking(particle_type=0, tasks=[search_task, rotation_task])

model = srl.networks.FlaxModel(
    flax_model=ActorCriticNetwork(),
    optimizer=optax.adam(learning_rate=0.0001),
    input_shape=(5, 2),
    sampling_strategy=sampling_strategy,
    exploration_policy=exploration_policy,
)

translate = Action(force=10.0)
rotate_clockwise = Action(torque=np.array([0.0, 0.0, 10.0])) 
rotate_counter_clockwise = Action(torque=np.array([0.0, 0.0, -10.0]))
do_nothing = Action(torque=np.array([0.0, 0.0, 10.0]))

actions = {
    "RotateClockwise": rotate_clockwise,
    "Translate": translate,
    "RotateCounterClockwise": rotate_counter_clockwise,
    "DoNothing": do_nothing,
}

system_runner = get_engine(system)

find_and_rotate.initialize(system_runner.colloids)
observable.initialize(system_runner.colloids)

ac_agent = srl.agents.ActorCriticAgent(
    particle_type=0,
    network=model,
    task=rotation_task, 
    observable=observable, 
    actions=actions,
    loss=loss
)

ac_agent.restore_agent("Models")

force_fn = srl.force_functions.ForceFunction({"0": ac_agent})

# Run the simulation

system_runner.integrate(5000, force_fn)
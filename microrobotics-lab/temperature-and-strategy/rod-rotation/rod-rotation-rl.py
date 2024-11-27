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
def get_engine(system_runner, cycle_index: str = "0"):
    # Simulation parameters
    seed = np.random.randint(645153513)

    temperature = TEMP
    n_colloids = 100

    ureg = pint.UnitRegistry()

    md_params = espresso.MDParams(
        ureg=ureg,
        fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
        WCA_epsilon=ureg.Quantity(1.0 + temperature, "kelvin") * ureg.boltzmann_constant,
        temperature=ureg.Quantity(temperature, "kelvin"),
        box_length=ureg.Quantity([150] * 3, "micrometer"),
        time_slice=ureg.Quantity(10.0, "second"),  # model timestep
        time_step=ureg.Quantity(0.01, "second"),  # integrator timestep
        write_interval=ureg.Quantity(2000000, "second"),
        periodic=False,
    )

    system_runner = srl.espresso.EspressoMD(
        md_params=md_params,
        n_dims=2,
        seed=seed,
        out_folder=f'./training',
        write_chunk_size=100,
        system=system_runner,
        h5_group_tag=cycle_index
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


n_episodes = 500
episode_length = 360

# Exploration policy
exploration_policy = srl.exploration_policies.RandomExploration(probability=0.0)

# Sampling strategy
sampling_strategy = srl.sampling_strategies.GumbelDistribution()

# Loss function
value_function = srl.value_functions.GAE(gamma=0.99, lambda_=0.95)
loss = srl.losses.ProximalPolicyLoss(
    entropy_coefficient=0.02, epsilon=0.2
    )

observable = srl.observables.SubdividedVisionCones(
    vision_range=1000000.0,
    vision_half_angle=1.4,
    n_cones=5,
    radii=jnp.array([3.08 for _ in range(100)] + [0.84745763 for _ in range(59)])
)

rotation_task = srl.tasks.object_movement.RotateRod(
    particle_type=0,
    angular_velocity_scale=1,
    rod_type=1,
    direction="CCW",
    partition=True,
    velocity_history=100
)

model = srl.networks.FlaxModel(
    flax_model=ActorCriticNetwork(),
    optimizer=optax.adamw(learning_rate=1e-4),
    input_shape=(5, 2),
    sampling_strategy=sampling_strategy,
    exploration_policy=exploration_policy,
)

translate = Action(force=20.0)
rotate_clockwise = Action(torque=np.array([0.0, 0.0, 10.0])) 
rotate_counter_clockwise = Action(torque=np.array([0.0, 0.0, -10.0]))
do_nothing = Action()

actions = {
    "RotateClockwise": rotate_clockwise,
    "Translate": translate,
    "RotateCounterClockwise": rotate_counter_clockwise,
    "DoNothing": do_nothing,
}

ac_agent = srl.agents.ActorCriticAgent(
    particle_type=0,
    network=model,
    task=rotation_task, 
    observable=observable, 
    actions=actions,
    loss=loss
)

rl_trainer = srl.trainers.EpisodicTrainer(
    [ac_agent],
)
# rl_trainer.restore_models()

rewards = rl_trainer.perform_rl_training(
    get_engine=get_engine,
    n_episodes=n_episodes,
    system=system,
    reset_frequency=1,
    episode_length=episode_length,
)

np.save("rewards.npy", rewards)

rl_trainer.export_models()

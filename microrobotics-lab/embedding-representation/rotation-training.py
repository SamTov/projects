# Oryx runs with GPU 1
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

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

    temperature = 0.0
    n_colloids = 50

    ureg = pint.UnitRegistry()

    md_params = espresso.MDParams(
        ureg=ureg,
        fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
        WCA_epsilon=ureg.Quantity(1.0, "kelvin") * ureg.boltzmann_constant,
        temperature=ureg.Quantity(temperature, "kelvin"),
        box_length=ureg.Quantity(150, "micrometer"),
        time_slice=ureg.Quantity(0.1, "second"),  # model timestep
        time_step=ureg.Quantity(0.001, "second"),  # integrator timestep
        write_interval=ureg.Quantity(2, "second"),
    )

    system_runner = srl.espresso.EspressoMD(
        md_params=md_params,
        n_dims=2,
        seed=seed,
        out_folder=f'./ep_training/{seed}/training',
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

# Reinforcement learning

## Models

class ColloidEmbedding(nn.Module):

    def setup(self):
        self.kernel_init = nn.initializers.xavier_normal()

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            12, 
            kernel_init=self.kernel_init, 
            )(x)
        x = nn.leaky_relu(x)
        return nn.Dense(
            features=5, 
            kernel_init=self.kernel_init, 
            )(x)
    
class RodEmbedding(nn.Module):

    def setup(self):
        self.kernel_init = nn.initializers.xavier_normal()

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            12, 
            kernel_init=self.kernel_init, 
            )(x)
        x = nn.leaky_relu(x)
        return nn.Dense(
            features=5, 
            kernel_init=self.kernel_init, 
            )(x)
    
class ActorCriticNetwork(nn.Module):
    """A simple AC network."""
    
    def setup(self):
        self.colloid_embedding = ColloidEmbedding()
        self.rod_embeding = RodEmbedding()

        self.kernel_init = nn.initializers.xavier_normal()
    
    @nn.compact
    def __call__(self, x):
        # Pass vision cone observations through the embeddings
        colloid_embedding = self.colloid_embedding(x[:, 0])
        rod_embedding = self.rod_embeding(x[:, 1])
        
        # Concatenate the embeddings
        vision_embedding = jnp.concatenate([colloid_embedding, rod_embedding], axis=-1)

        # Actor pass
        x = nn.Dense(
            features=12, 
            kernel_init=self.kernel_init, 
            )(vision_embedding)
        x = nn.leaky_relu(x)
        x = nn.Dense(
            features=12, 
            kernel_init=self.kernel_init, 
            )(x)
        x = nn.leaky_relu(x)

        # Critic pass
        y = nn.Dense(
            features=12, 
            kernel_init=self.kernel_init, 
            )(vision_embedding)
        y = nn.leaky_relu(y)
        y = nn.Dense(
            features=12, 
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


n_episodes = 10000
episode_length = 100

# Exploration policy
exploration_policy = srl.exploration_policies.RandomExploration(probability=0.0)

# Sampling strategy
sampling_strategy = srl.sampling_strategies.GumbelDistribution()

# Loss function
value_function = srl.value_functions.GAE(gamma=0.99, lambda_=0.95)
loss = srl.losses.ProximalPolicyLoss(entropy_coefficient=0.001, epsilon=0.1)

observable = srl.observables.SubdividedVisionCones(
    vision_range=1000000.0,
    vision_half_angle=1.4,
    n_cones=5,
    radii=jnp.array([3.08 for _ in range(50)] + [0.84745763 for _ in range(59)])
)

rotation_task = srl.tasks.object_movement.RotateRod(
    particle_type=0,
    angular_velocity_scale=100,
    rod_type=1,
    direction="CCW",
    partition=True,
    velocity_history=100
)

def decay_fn(x):
    return 1.0 - x

search_task = srl.tasks.searching.SpeciesSearch(
        decay_fn=decay_fn,
        box_length = np.array([1000.0, 1000.0, 1000.0]),
        sensing_type = 1,
        avoid = False,
        scale_factor = 10,
        particle_type = 0,
)

find_and_rotate = srl.tasks.MultiTasking(particle_type=0, tasks=[search_task, rotation_task])

model = srl.networks.FlaxModel(
    flax_model=ActorCriticNetwork(),
    optimizer=optax.adam(learning_rate=0.002),
    input_shape=(5, 2),
    sampling_strategy=sampling_strategy,
    exploration_policy=exploration_policy,
)
# Restore state before continuing training
# model.restore_model_state(
#     filename="Model0", directory="Models/"
#     )

translate = Action(force=50.0)
rotate_clockwise = Action(torque=np.array([0.0, 0.0, 10.0])) 
rotate_counter_clockwise = Action(torque=np.array([0.0, 0.0, -10.0]))
do_nothing = Action(torque=np.array([0.0, 0.0, 10.0]))

actions = {
    "RotateClockwise": rotate_clockwise,
    "Translate": translate,
    "RotateCounterClockwise": rotate_counter_clockwise,
    "DoNothing": do_nothing,
}

ac_agent = srl.agents.ActorCriticAgent(
    particle_type=0,
    network=model,
    task=find_and_rotate, 
    observable=observable, 
    actions=actions
)

rl_trainer = srl.trainers.EpisodicTrainer(
    [ac_agent],
    loss,
)

rewards = rl_trainer.perform_rl_training(
    get_engine=get_engine,
    n_episodes=n_episodes,
    system=system,
    reset_frequency=10,
    episode_length=episode_length,
            )
np.save("exploration.npy", rewards)

rl_trainer.export_models()

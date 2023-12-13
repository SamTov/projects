# SwarmRL Imports
import swarmrl as srl
import swarmrl.engine.espresso as espresso
from swarmrl.actions import Action

# espresso imports
import espressomd

# Linalg imports
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import optax

# Helper imports
import pint


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
        time_slice=ureg.Quantity(0.5, "second"),  # model timestep
        time_step=ureg.Quantity(0.001, "second"),  # integrator timestep
        write_interval=ureg.Quantity(2, "second"),
    )

    system_runner = srl.espresso.EspressoMD(
        md_params=md_params,
        n_dims=2,
        seed=seed,
        out_folder=f'./deployment/',
        write_chunk_size=100,
        system=system_runner,
        periodic=False
    )

    system_runner.add_colloids(
        n_colloids,
        ureg.Quantity(3.08, "micrometer"),
        ureg.Quantity(np.array([75., 75., 0]), "micrometer"),
        ureg.Quantity(50, "micrometer"),
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
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(12)(x)
        x = nn.relu(x)
        return nn.Dense(features=5)(x)
    
class RodEmbedding(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(12)(x)
        x = nn.relu(x)
        return nn.Dense(features=5)(x)
    
class ActorCriticNetwork(nn.Module):
    """A simple AC network."""
    
    def setup(self):
        self.colloid_embedding = ColloidEmbedding()
        self.rod_embeding = RodEmbedding()
    
    @nn.compact
    def __call__(self, x):
        # Pass vision cone observations through the embeddings
        colloid_embedding = self.colloid_embedding(x[:, 0])
        rod_embedding = self.rod_embeding(x[:, 1])
        
        # Concatenate the embeddings
        vision_embedding = jnp.concatenate([colloid_embedding, rod_embedding], axis=-1)

        # Actor pass
        x = nn.Dense(features=12)(vision_embedding)
        x = nn.relu(x)
        x = nn.Dense(features=12)(x)
        x = nn.relu(x)

        # Critic pass
        y = nn.Dense(features=12)(vision_embedding)
        y = nn.relu(y)
        y = nn.Dense(features=12)(y)
        y = nn.relu(y)

        # Actor head
        actions = nn.Dense(features=4)(x)

        # Critic head
        value = nn.Dense(features=1)(y)

        return actions, value



n_episodes = 5000
episode_length = 20

# Exploration policy
exploration_policy = srl.exploration_policies.RandomExploration(probability=0.0)

# Sampling strategy
sampling_strategy = srl.sampling_strategies.GumbelDistribution()

# Loss function
loss = srl.losses.PolicyGradientLoss()

observable = srl.observables.SubdividedVisionCones(
    vision_range=1000000.0,
    vision_half_angle=1.4,
    n_cones=5,
    radii=jnp.array([3.08 for _ in range(50)] + [0.84745763 for _ in range(59)])
)

rotation_task = srl.tasks.object_movement.RotateRod(
    particle_type=0,
    angular_velocity_scale=10000,
    rod_type=1,
    direction="CCW",
    partition=True,
    velocity_history=50
)

model = srl.networks.FlaxModel(
    flax_model=ActorCriticNetwork(),
    optimizer=optax.adam(learning_rate=0.001),
    input_shape=(5, 2),
    sampling_strategy=sampling_strategy,
    exploration_policy=exploration_policy,
)
# Restore state before continuing training
model.restore_model_state(
    filename="Model0", directory="Models/"
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

ac_agent = srl.agents.ActorCriticAgent(
    particle_type=0,
    network=model,
    task=rotation_task, 
    observable=observable, 
    actions=actions
)

force_fn = srl.force_functions.ForceFunction({"0": ac_agent})

system_runner = get_engine(system)

rotation_task.initialize(system_runner.colloids)

system_runner.integrate(6000, force_fn)

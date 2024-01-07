# SwarmRL Imports
import swarmrl as srl
from swarmrl.components import Colloid

# Helper Imports
from typing import List

# Linalg Imports
import jax.numpy as np
import numpy as onp
import flax.linen as nn


def field_decay(distance):
    return 1 / distance

# Chemotaxis task
class ChemotaxisTask(srl.tasks.Task):
    """
    Custom chemotaxis task for the agents.
    """
    def __init__(
        self,
        source: np.ndarray = np.array([0, 0, 0]),
        decay_function: callable = None,
        box_length: np.ndarray = np.array([1.0, 1.0, 0.0]),
        reward_scale_factor: int = 10,
        particle_type: int = 0,
    ):
        """
        Constructor for the find origin task.

        Parameters
        ----------
        source : np.ndarray (default = (0, 0 0))
                Source of the gradient.
        decay_function : callable (required=True)
                A function that describes the decay of the field along one dimension.
                This cannot be left None. The function should take a distance from the
                source and return the magnitude of the field at this point.
        box_length : np.ndarray
                Side length of the box.
        reward_scale_factor : int (default=10)
                The amount the field is scaled by to get the reward.
        particle_type : int (default=0)

        """
        super().__init__(particle_type=particle_type)
        self.source = source / box_length
        self.decay_fn = decay_function
        self.reward_scale_factor = reward_scale_factor
        self.box_length = box_length

    def change_source(self, new_source: np.ndarray):
        """
        Changes the concentration field source.

        Parameters
        ----------
        new_source : np.ndarray
                Coordinates of the new source.
        """
        self.source = new_source

    def compute_colloid_reward(self, index: int, colloids):
        """
        Compute the reward for a single colloid.

        Parameters
        ----------
        index : int
                Index of the colloid to compute the reward for.

        Returns
        -------
        reward : float
                Reward for the colloid.
        """
        # Get the current position of the colloid
        position = onp.copy(colloids[index].pos) / self.box_length

        # Compute the field value
        field_value = self.decay_fn(
            np.linalg.norm(position - self.source)
        )

        # Compute the reward
        reward = np.clip(self.reward_scale_factor * field_value, 0.0, None)

        return reward

    def __call__(self, colloids: List[Colloid]):
        """
        Compute the reward.

        In this case of this task, the observable itself is the gradient of the field
        that the colloid is swimming in. Therefore, the change is simply scaled and
        returned.

        Parameters
        ----------
        colloids : List[Colloid] (n_colloids, )
                List of colloids to be used in the task.

        Returns
        -------
        rewards : List[float] (n_colloids, )
                Rewards for each colloid.
        """
        colloid_indices = self.get_colloid_indices(colloids)

        return np.array(
            [self.compute_colloid_reward(index, colloids) for index in colloid_indices]
        )
    

class ChemotaxisObservable(srl.observables.Observable):
    """
    Observable for the chemotaxis problem.
    """

    def __init__(
        self,
        source: np.ndarray,
        decay_fn: callable,
        box_length: np.ndarray,
        scale_factor: int = 100,
        particle_type: int = 0,
    ):
        """
        Constructor for the observable.

        Parameters
        ----------
        source : np.ndarray
                Source of the field.
        decay_fn : callable
                Decay function of the field.
        box_size : np.ndarray
                Array for scaling of the distances.
        scale_factor : int (default=100)
                Scaling factor for the observable.
        particle_type : int (default=0)
                Particle type to compute the observable for.
        """
        super().__init__(particle_type=particle_type)

        self.source = source / box_length
        self.decay_fn = decay_fn
        self.box_length = box_length
        self.scale_factor = scale_factor
        self._observable_shape = (3,)

    def compute_single_observable(self, index: int, colloids: List[Colloid]) -> float:
        """
        Compute the observable for a single colloid.

        Parameters
        ----------
        index : int
                Index of the colloid to compute the observable for.
        colloids : List[Colloid]
                List of colloids in the system.
        """
        reference_colloid = colloids[index]
        position = onp.copy(reference_colloid.pos) / self.box_length
        index = onp.copy(reference_colloid.id)

        field_value = self.decay_fn(
            np.linalg.norm(self.source - position)
        )

        return self.scale_factor * field_value

    def compute_observable(self, colloids: List[Colloid]):
        """
        Compute the position of the colloid.

        Parameters
        ----------
        colloids : List[Colloid] (n_colloids, )
                List of all colloids in the system.

        Returns
        -------
        observables : List[float] (n_colloids, dimension)
                List of observables, one for each colloid. In this case,
                current field value minus to previous field value.
        """
        reference_ids = self.get_colloid_indices(colloids)

        observables = [
            self.compute_single_observable(index, colloids) for index in reference_ids
        ]

        return np.array(observables).reshape(-1, 1)
    
class ActorCriticNet(nn.Module):
    """A simple dense model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        y = nn.Dense(features=1)(x)
        x = nn.Dense(features=4)(x)
        return x, y

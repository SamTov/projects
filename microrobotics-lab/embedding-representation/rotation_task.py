# SwarmRL imports
import swarmrl as srl
from swarmrl.actions import Action
from swarmrl.components import Colloid
from swarmrl.utils.colloid_utils import compute_torque_partition_on_rod

# Structure imports
from typing import List

# Linalg imports
import numpy as onp
import jax.numpy as np
import jax


@jax.jit
def decay_fn(x):
    return 1.0 - x


class RotationTask(srl.tasks.Task):
    """
    SwamRL task for rotation study.
    """
    def __init__(
            self,
            rod_type: int,
            particle_type: int,
            direction: str,
            sensing_type: int,
            box_length: np.ndarray,
            running_average_window: int = 10,
    ):
        """
        Constructor for the task.

        Parameters
        ----------
        rod_type : int
            Rod type.
        particle_type : int
            Particle type.
        direction : str
            Direction of rotation.
        running_average_window : int
            Window for running average.
        """
        super().__init__(particle_type=particle_type)

        self.rod_type = rod_type
        self.particle_type = particle_type
        self.direction = direction
        self.running_average_window = running_average_window
        self.box_length = box_length
        self.sensing_type = sensing_type
        self.velocity_history = running_average_window

        # Rod stuff
        self._velocity_history = np.zeros(running_average_window)
        self._append_index = int(running_average_window - 1)
        self.decomp_fn = jax.jit(compute_torque_partition_on_rod)

        # Sensing stuff
        self.historical_field = {}
        self.task_fn = jax.vmap(
            self.compute_single_particle_task, in_axes=(0, 0, None, 0)
        )
        self.decay_fn = decay_fn

    def _initialize_rod(self, colloids: List[Colloid]):
        """
        Initialize the rod part of the task.
        """
        self._velocity_history = np.zeros(self.velocity_history)
        self._append_index = int(self.velocity_history - 1)
        for item in colloids:
            if item.type == self.rod_type:
                self._historic_rod_director = onp.copy(item.director)
                break

    def _initialize_sensing(self, colloids: List[Colloid]):
        """
        Initialize the sensing part of the task.
        """
        reference_ids = self.get_colloid_indices(colloids)
        historic_values = np.zeros(len(reference_ids))

        positions = []
        indices = []
        for index in reference_ids:
            indices.append(colloids[index].id)
            positions.append(colloids[index].pos)

        test_points = np.array(
            [colloid.pos for colloid in colloids if colloid.type == self.sensing_type]
        )

        out_indices, _, field_values = self.task_fn(
            np.array(indices), np.array(positions), test_points, historic_values
        )

        for index, value in zip(out_indices, onp.array(field_values)):
            self.historical_field[str(index)] = value

    def initialize(self, colloids: List[Colloid]):
        """
        Prepare the task for running.

        In this case, as all rod directors are the same, we
        only need to take on for the historical value.

        Parameters
        ----------
        colloids : List[Colloid]
                List of colloids to be used in the task.

        Returns
        -------
        Updates the class state.
        """
        self._initialize_rod(colloids)
        self._initialize_sensing(colloids)

    def compute_single_particle_task(
        self,
        index: int,
        reference_position: np.ndarray,
        test_positions: np.ndarray,
        historic_value: float,
    ) -> tuple:
        """
        Compute the task for a single colloid.

        Parameters
        ----------
        index : int
                Index of the colloid to compute the observable for.
        reference_position : np.ndarray (3,)
                Position of the reference colloid.
        test_positions : np.ndarray (n_colloids, 3)
                Positions of the test colloids.
        historic_value : float
                Historic value of the observable.

        Returns
        -------
        tuple (index, task_value)
        index : int
                Index of the colloid to compute the observable for.
        task_value : float
                Value of the task.
        """
        distances = np.linalg.norm(
            (test_positions - reference_position) / self.box_length, axis=-1
        )
        indices = np.asarray(np.nonzero(distances, size=distances.shape[0] - 1))
        distances = np.take(distances, indices, axis=0)
        field_value = np.max(self.decay_fn(distances))

        return index, field_value, field_value
    
    def compute_sensing_task(self, colloids: List[Colloid]):
        """
        Compute the sensing part of the task.
        """
        reference_ids = self.get_colloid_indices(colloids)
        positions = []
        indices = []
        historic_values = []
        for index in reference_ids:
            indices.append(colloids[index].id)
            positions.append(colloids[index].pos)
            historic_values.append(self.historical_field[str(colloids[index].id)])

        test_points = np.array(
            [colloid.pos for colloid in colloids if colloid.type == self.sensing_type]
        )

        out_indices, delta_values, field_values = self.task_fn(
            np.array(indices),
            np.array(positions),
            test_points,
            np.array(historic_values),
        )

        for index, value in zip(out_indices, onp.array(field_values)):
            self.historical_field[str(index)] = value

        return np.clip(delta_values, 0.0, None)
        

    def _compute_angular_velocity(self, new_director: np.ndarray):
        """
        Compute the instantaneous angular velocity of the rod.

        Parameters
        ----------
        new_director : np.ndarray (3, )
                New rod director.

        Returns
        -------
        angular_velocity : float
                Angular velocity of the rod
        """
        angular_velocity = np.arctan2(
            np.cross(self._historic_rod_director[:2], new_director[:2]),
            np.dot(self._historic_rod_director[:2], new_director[:2]),
        )

        # Convert to rph for better scaling.
        angular_velocity = 3600 *  (np.rad2deg(angular_velocity) / (0.1 * 360.0))
        
        # Update the historical rod director and velocity.
        self._historic_rod_director = new_director
        self._velocity_history = np.roll(self._velocity_history, -1)
        self._velocity_history = self._velocity_history.at[self._append_index].set(
            angular_velocity
        )

        # Return the scaled average velocity.
        return np.clip(
            np.nanmean(self._velocity_history), 0.0, None
        )

    def partition_reward(
        self,
        reward: float,
        colloid_positions: np.ndarray,
        colloid_directors: np.ndarray,
        rod_positions: np.ndarray,
        rod_directors: np.ndarray,
    ) -> np.ndarray:
        """
        Partition a reward into colloid contributions.

        Parameters
        ----------
        reward : float
                Reward to be partitioned.
        colloid_positions : np.ndarray (n_colloids, 3)
                Positions of the colloids.
        colloid_directors : np.ndarray (n_colloids, 3)
                Directors of the colloids.
        rod_positions : np.ndarray (n_rod, 3)
                Positions of the rod particles.
        rod_directors : np.ndarray (n_rod, 3)
                Directors of the rod particles.

        Returns
        -------
        partitioned_reward : np.ndarray (n_colloids, )
                Partitioned reward for each colloid.
        """
        colloid_partitions = self.decomp_fn(
            colloid_positions, colloid_directors, rod_positions, rod_directors
        )

        return reward * colloid_partitions

    def _compute_angular_velocity_reward(
        self,
        rod_directors: np.ndarray,
        rod_positions: np.ndarray,
        colloid_positions: np.ndarray,
        colloid_directors: np.ndarray,
    ):
        """
        Compute the angular velocity reward.

        Parameters
        ----------
        rod_directors : np.ndarray (n_rod, 3)
                Directors of the rod particles.
        rod_positions : np.ndarray (n_rod, 3)
                Positions of the rod particles.
        colloid_positions : np.ndarray (n_colloids, 3)
                Positions of the colloids.
        colloid_directors : np.ndarray (n_colloids, 3)
                Directors of the colloids.

        Returns
        -------
        angular_velocity_reward : float
                Angular velocity reward.
        """
        # Compute angular velocity
        angular_velocity = self._compute_angular_velocity(rod_directors[0])
        # Compute colloid-wise rewards
        return self.partition_reward(
            angular_velocity,
            colloid_positions,
            colloid_directors,
            rod_positions,
            rod_directors,
        )
    
    def compute_rotation_reward(self, colloids: List[Colloid]):
        """
        Compute the rod componant of the reward.
        """
        # Collect the important data.
        rod = [colloid for colloid in colloids if colloid.type == self.rod_type]
        rod_positions = np.array([colloid.pos for colloid in rod])
        rod_directors = np.array([colloid.director for colloid in rod])

        chosen_colloids = [
            colloid for colloid in colloids if colloid.type == self.particle_type
        ]
        colloid_positions = np.array([colloid.pos for colloid in chosen_colloids])
        colloid_directors = np.array([colloid.director for colloid in chosen_colloids])

        # Compute the angular velocity reward
        angular_velocity_term = self._compute_angular_velocity_reward(
            rod_directors, rod_positions, colloid_positions, colloid_directors
        )

        return angular_velocity_term

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
        angular_velocity_term = self.compute_rotation_reward(colloids)
        sensing_term = self.compute_sensing_task(colloids)

        return angular_velocity_term + sensing_term

from collections.abc import Callable

import numpy as np


class Particle:
    def __init__(self, num_dims: int, lower_bound: float, upper_bound: float) -> None:
        """Initialize the particle.

        Args:
            num_dims (int): Number of dimensions
            lower_bound (float): Lower bound of the search space
            upper_bound (float): Upper bound of the search space
        """

        # Current position, velocity and value of the particle.
        self.position: np.ndarray = np.random.uniform(
            lower_bound, upper_bound, num_dims
        )
        self.velocity: np.ndarray = np.zeros(num_dims)
        self.value: np.float64 = np.float64(np.inf)

        # Personal best position and value.
        self.personal_best_position: np.ndarray = np.full(
            shape=(num_dims,), fill_value=np.inf
        )
        self.personal_best_value: np.float64 = np.float64(np.inf)

        return

    def evaluate(self, func: Callable[[np.ndarray], np.float64]) -> None:
        """Evaluate the current position using the given objective function.

        Args:
            func (Callable[[np.ndarray], np.float64]): Objective function to evaluate the particle's position
        """

        self.value = func(self.position)

        return

    def update_velocity(
        self, weight: float, c1: float, c2: float, global_best_position: np.ndarray
    ) -> None:
        """Update the velocity of the particle.

        Args:
            weight (float): Inertia weight
            c1 (float): Cognitive parameter
            c2 (float): Social parameter
            global_best_position (np.ndarray): Current global best position
        """

        r1: np.ndarray = np.random.rand(len(self.position))
        r2: np.ndarray = np.random.rand(len(self.position))

        cognitive_component: np.ndarray = (
            c1 * r1 * (self.personal_best_position - self.position)
        )
        social_component: np.ndarray = c2 * r2 * (global_best_position - self.position)

        self.velocity = weight * self.velocity + cognitive_component + social_component

        return

    def update_position(self) -> None:
        """Update the position of the particle."""

        self.position += self.velocity

        return

    def update_personal_best(self) -> None:
        """Update the personal best position and value."""

        if self.value < self.personal_best_value:
            # Update personal best position and value if the current value is better than the personal best value.
            self.personal_best_position = self.position.copy()
            self.personal_best_value = self.value

        return

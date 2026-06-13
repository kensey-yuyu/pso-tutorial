import numpy as np
from tqdm import tqdm

from pso.functions.functions import Functions
from pso.particle import Particle
from utils.logger import Logger


class PSO:
    def __init__(
        self,
        num_particles: int,
        weight: float,
        c1: float,
        c2: float,
        func_num: int,
        num_dims: int,
        lower_bound: float,
        upper_bound: float,
        iterations: int,
        logger: Logger,
    ) -> None:
        """Initialize the Particle Swarm Optimization.

        Args:
            num_particles (int): Number of particles
            weight (float): Inertia weight
            c1 (float): Cognitive parameter
            c2 (float): Social parameter
            func_num (int): Function number to optimize
            num_dims (int): Number of dimensions
            lower_bound (float): Lower bound of the search space
            upper_bound (float): Upper bound of the search space
            iterations (int): Number of iterations
            logger (Logger): Logger instance for logging optimization progress
        """

        self.num_particles: int = num_particles
        self.weight: float = weight
        self.c1: float = c1
        self.c2: float = c2
        self.func_num: int = func_num
        self.num_dims: int = num_dims
        self.lower_bound: float = lower_bound
        self.upper_bound: float = upper_bound
        self.iterations: int = iterations
        self.logger: Logger = logger

        # Functions are initialized.
        self.functions = Functions(func_num=func_num, num_dims=num_dims)

        # Particles are initialized.
        self.particles: list[Particle] = [
            Particle(
                num_dims=num_dims, lower_bound=lower_bound, upper_bound=upper_bound
            )
            for _ in range(num_particles)
        ]

        # Global best position and value.
        self.global_best_position: np.ndarray = np.full(
            shape=(num_dims,), fill_value=np.inf
        )
        self.global_best_value: np.float64 = np.float64(np.inf)

        return

    def optimize(self) -> None:
        """Optimize the function using Particle Swarm Optimization."""

        # Set the initial positions and values of the particles as their personal bests.
        for particle in self.particles:
            particle.evaluate(func=self.functions.get_func())
            particle.update_personal_best()
        self.update_global_best()
        self.logger.log(
            iteration=0,
            particles=self.particles,
            global_best_position=self.global_best_position,
            global_best_value=self.global_best_value,
        )
        print(f"Initial Global Best Value: {self.global_best_value}")

        # Optimize the function for the specified number of iterations.
        for iter in tqdm(range(self.iterations), desc="Optimizing"):
            # Update the velocities.
            for particle in self.particles:
                particle.update_velocity(
                    weight=self.weight,
                    c1=self.c1,
                    c2=self.c2,
                    global_best_position=self.global_best_position,
                )

            # Update the positions.
            for particle in self.particles:
                particle.update_position()

            # Evaluate the current positions.
            for particle in self.particles:
                particle.evaluate(func=self.functions.get_func())

            # Update the personal best positions and values.
            for particle in self.particles:
                particle.update_personal_best()

            # Update the global best position and value.
            self.update_global_best()

            # Log the optimization progress.
            self.logger.log(
                iteration=iter + 1,
                particles=self.particles,
                global_best_position=self.global_best_position,
                global_best_value=self.global_best_value,
            )

            # FIXME: format the iteration number and global best value.
            tqdm.write(
                f"Iteration {iter + 1}/{self.iterations}, Global Best Value: {self.global_best_value}"
            )

        return

    def update_global_best(self) -> None:
        """Update the global best position and value."""

        for particle in self.particles:
            if particle.personal_best_value < self.global_best_value:
                # Update the global best position and value if the current particle's personal best is better.
                self.global_best_position = particle.personal_best_position.copy()
                self.global_best_value = particle.personal_best_value

        return

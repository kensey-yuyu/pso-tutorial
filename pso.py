import numpy as np
from tqdm import tqdm

from particle import Particle


class PSO:
    """PSO.

    Attributes:
        n (int): The number of particles.
        particles (list[Particle]): The particles, its number is n.
        global_best_fitness (float): The global best fitness.
        global_best_position (float): The global best position.
        history (dict[str, list[float | np.ndarray]]): The history of global best.
        function (): The evaluation function.
    """

    def __init__(
        self,
        n: int,
        iterations: int,
        dim: int,
        upper: float,
        lower: float,
        limit_init_velocity: float,
        weight: float,
        c1: float,
        c2: float,
        function
    ) -> None:
        self.iterations: int = iterations
        # Particles.
        self.particles: list[Particle] = [Particle(
            dim, upper, lower, limit_init_velocity, weight, c1, c2) for _ in range(n)]
        # Global bests.
        self.global_best_fitness: float = np.inf
        self.global_best_position: np.ndarray = np.full(dim, np.inf)
        self.history: dict[str, list[float | np.ndarray]] = {
            "global_best_fitnesses": [],
            "global_best_positions": []
        }

        # Evaluation function.
        self.function = function

        # Initial evaluation.
        self.evaluate()
        self.update_personal_best()
        self.update_global_best()
        return

    def optimize(self) -> None:
        """Optimize particle positions to get global optimum.
        """

        # NOTE: If the initial evaluation (initialization) is excluded from iterations, start from 0.
        for iter in tqdm(range(1, self.iterations)):
            self.update_particles()
            self.evaluate()
            self.update_personal_best()
            self.update_global_best()
            tqdm.write(
                f"Iter {iter:{len(str(self.iterations))}}, Fitness: {self.global_best_fitness:.5e}")
        return

    def evaluate(self) -> None:
        """Evaluate each particle position by applying function, in other words, calculate fitnesses.
        """

        for particle in self.particles:
            particle.calculate_fitness(self.function)
        return

    def update_particles(self) -> None:
        """Update each particle position.
        """

        for particle in self.particles:
            # Update velocity.
            particle.update_velocity(self.global_best_position)
            # Update position.
            particle.update_position()
        return

    def update_personal_best(self) -> None:
        """Update each particle personal best.
        """

        for particle in self.particles:
            particle.update_personal_best()
        return

    def update_global_best(self) -> None:
        """Update global best.
        """

        for particle in self.particles:
            if particle.personal_best_fitness < self.global_best_fitness:
                self.global_best_fitness = particle.personal_best_fitness
                self.global_best_position = particle.personal_best_position.copy()
        self.history["global_best_fitnesses"].append(
            self.global_best_fitness)
        self.history["global_best_positions"].append(
            self.global_best_position.copy())
        return

    def get_history(self) -> dict[str, np.ndarray]:
        """Get history of PSO exploration.

        Returns:
            dict[str, dict[str, np.ndarray]: The history contains records of global best and all particles.
        """

        history: dict[str, np.ndarray] = {
            "global_best_fitnesses": np.stack(self.history["global_best_fitnesses"]),
            "global_best_positions": np.stack(self.history["global_best_positions"]),
            "personal_best_fitnesses": np.stack([p.history["personal_best_fitnesses"] for p in self.particles]),
            "personal_best_positions": np.stack([p.history["personal_best_positions"] for p in self.particles]),
            "fitnesses": np.stack([p.history["fitnesses"] for p in self.particles]),
            "positions": np.stack([p.history["positions"] for p in self.particles]),
            "velocities": np.stack([p.history["velocities"] for p in self.particles])
        }
        return history

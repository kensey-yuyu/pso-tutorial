import numpy as np


class Particle:
    """Particle.

    Attributes:
        dim (int): The number of dimension.
        limit_init_velocity (float): Limit of initial velocity.
        weight (float): Weight of velocity. It is hyperparameter.
        c1 (float): Weight of personal best. It is hyperparameter.
        c2 (float): Weight of global best. It is hyperparameter.
        upper (float): Upper boundary of domain search.
        lower (float): Lower boundary of domain search.
        fitness (float): The fitness.
        position (np.ndarray): The position.
        velocity (np.ndarray): The velocity.
        personal_best_fitness: The personal best fitness.
        personal_best_position: The personal best position.
        history (dict[str, list[float | np.ndarray]]): The history of particle.
    """

    def __init__(
        self,
        dim: int,
        upper: float,
        lower: float,
        limit_init_velocity: float,
        weight: float,
        c1: float,
        c2: float
    ) -> None:
        self.dim: int = dim
        self.weight: float = weight
        self.c1: float = c1
        self.c2: float = c2
        self.upper: float = upper
        self.lower: float = lower
        self.fitness: float = np.inf
        self.position: np.ndarray = np.random.uniform(lower, upper, dim)
        self.velocity: np.ndarray = np.random.uniform(-limit_init_velocity,
                                                      limit_init_velocity, dim)
        self.personal_best_fitness: float = np.inf
        self.personal_best_position: np.ndarray = np.full(dim, np.inf)
        self.history: dict[str, list[float | np.ndarray]] = {
            "fitnesses": [],
            "positions": [],
            "velocities": [],
            "personal_best_fitnesses": [],
            "personal_best_positions": []
        }
        self.history["positions"].append(self.position.copy())
        self.history["velocities"].append(self.velocity.copy())
        return

    def calculate_fitness(self, function) -> None:
        """Calculate fitness with current position.

        Args:
            function (): Evaluation function.
        """

        # Check if position is outside lower and upper bounds.
        if (self.lower <= self.position).all() and (self.position <= self.upper).all():
            self.fitness = function(self.position)
        else:
            # If position is outside lower and upper bounds, fitness is invalid.
            self.fitness = np.nan
        self.history["fitnesses"].append(self.fitness)
        return

    def update_velocity(self, global_best_position: np.ndarray) -> None:
        """Update velocity.

        Args:
            global_best_position (np.ndarray): Global best position.
        """

        # NOTE: In this program, the random numbers are generated for each dimension.
        self.velocity = self.weight * self.velocity + \
            self.c1 * np.random.rand(*self.velocity.shape) * \
            (self.personal_best_position - self.position) + \
            self.c2 * np.random.rand(*self.velocity.shape) * \
            (global_best_position - self.position)
        self.history["velocities"].append(self.velocity.copy())
        return

    def update_position(self) -> None:
        """Update position.
        """

        self.position = self.position + self.velocity
        self.history["positions"].append(self.position.copy())
        return

    def update_personal_best(self) -> None:
        """Update personal best.
        """

        if self.fitness < self.personal_best_fitness:
            self.personal_best_fitness = self.fitness
            self.personal_best_position = self.position.copy()
        self.history["personal_best_fitnesses"].append(
            self.personal_best_fitness)
        self.history["personal_best_positions"].append(
            self.personal_best_position.copy())
        return

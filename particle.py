import numpy as np


class Particle:
    """Particle.

    Args:
        dim (int): Dimension.
        upper (float): Upper boundary of domain search.
        lower (float): Lower boundary of domain search.
        limit_init_speed (float): Limit of initial speed.
        weight (float): Weight for update speed.
        c1 (float): Hyper parameter for bias of global best.     
        c2 (float): Hyper parameter for bias of personal best.    
    """

    def __init__(
        self,
        dim,
        upper,
        lower,
        limit_init_speed,
        weight,
        c1,
        c2
    ) -> None:
        self.dim = dim
        self.weight = weight
        self.c1 = c1
        self.c2 = c2
        self.upper = upper
        self.lower = lower
        self.position = np.random.uniform(lower, upper, dim)
        self.speed = np.random.uniform(-limit_init_speed,
                                       limit_init_speed, dim)
        self.fitness = float("inf")
        self.personal_best = float("inf")
        self.personal_best_position = np.full(dim, float("inf"))
        self.history = {
            "position": [],
            "speed": [],
            "fitness": [],
            "personal_best": [],
            "personal_best_position": []
        }
        self.history["position"].append(self.position)
        self.history["speed"].append(self.speed)
        return

    def update_position(self) -> None:
        self.position = self.position + self.speed
        while ((self.position < self.lower).any() and (self.upper < self.position).any()):
            over = np.where(self.position > self.upper,
                            self.position - self.upper, 0)
            self.position = self.position - 2 * over
            under = np.where(self.position < self.lower,
                             self.lower - self.position, 0)
            self.position = self.position + 2 * under
        self.history["position"].append(self.position)
        return

    def update_speed(self, global_best_position) -> None:
        self.speed = self.weight * self.speed \
            + self.c1 * np.random.random(self.dim) * (global_best_position - self.position) \
            + self.c2 * np.random.random(self.dim) * \
            (self.personal_best_position - self.position)
        self.history["speed"].append(self.speed)
        return

    def update_fitness(self, fitness) -> None:
        self.fitness = fitness
        self.history["fitness"].append(self.fitness)
        return

    def update_personal_best(self) -> None:
        update = self.fitness < self.personal_best
        if update:
            self.personal_best = self.fitness
        self.update_personal_best_position(update)
        self.history["personal_best"].append(self.personal_best)
        return

    def update_personal_best_position(self, update) -> None:
        if update:
            self.personal_best_position = self.position
        self.history["personal_best_position"].append(
            self.personal_best_position)
        return

    def get_position(self) -> np.ndarray:
        return self.position

    def get_speed(self) -> np.ndarray:
        return self.speed

    def get_fitness(self) -> float:
        return self.fitness

    def get_personal_best(self) -> float:
        return self.personal_best

    def get_personal_best_position(self) -> np.ndarray:
        return self.personal_best_position

    def get_history(self) -> dict:
        return self.history

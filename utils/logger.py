import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt


class Logger:
    """Logger to save PSO result.

    Attributes:
        root (str): Root path for saving results.
        path (str): Path for saving this experiment result.
    """

    def __init__(self, dir_name: str = "") -> None:
        self.root: str = "./log"
        self.path: str = self.root + "/" +  \
            (dir_name or format(datetime.now(), "%Y-%m-%d_%H:%M:%S"))

        # Create log directory.
        os.makedirs(self.path, exist_ok=True)

        # TODO: Save configurations. (ex. hyperparameter)
        return

    def save(self, history: dict[str, np.ndarray]) -> None:
        """Save PSO result.

        Args:
            history (dict[str, np.ndarray]): All data of PSO exploration.
        """

        # Save all pso history.
        np.savez(f"{self.path}/data.npz", **history)

        # Plot global best fitnesses.
        plt.plot(history["global_best_fitnesses"])
        plt.title(
            f"Global best fitness: {history["global_best_fitnesses"][-1]:.5f}")
        plt.xlabel("Iterations")
        plt.ylabel("Fitness value")
        plt.savefig(f"{self.path}/fitness.svg", format="svg")
        plt.savefig(f"{self.path}/fitness_transparent.svg",
                    format="svg", transparent=True)
        # Plot log scale graph.
        plt.yscale("log")
        plt.savefig(f"{self.path}/fitness_log.svg", format="svg")
        plt.savefig(f"{self.path}/fitness_log_transparent.svg",
                    format="svg", transparent=True)
        plt.close()

        return

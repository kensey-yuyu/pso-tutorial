import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from pso.particle import Particle


class Logger:

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
    ) -> None:
        """Initialize the logger.

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
        """

        # Log data.
        self.log_data: dict[str, np.ndarray] = {
            "iteration": np.full(shape=(iterations + 1,), fill_value=np.nan),
            "positions": np.full(
                shape=(iterations + 1, num_particles, num_dims), fill_value=np.nan
            ),
            "values": np.full(shape=(iterations + 1, num_particles), fill_value=np.nan),
        }

        # Log for global best positions and values.
        self.global_best_log_data: dict[str, np.ndarray] = {
            "iteration": np.full(shape=(iterations + 1,), fill_value=np.nan),
            "position": np.full(shape=(iterations + 1, num_dims), fill_value=np.nan),
            "value": np.full(shape=(iterations + 1,), fill_value=np.nan),
        }

        # Path to save the log files and make the directory.
        self.path: str = f"results/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.path, exist_ok=True)

        # Save the parameters defined to the log file.
        with open(file=f"{self.path}/parameters.txt", mode="w") as f:
            f.write(f"num_particles: {num_particles}\n")
            f.write(f"weight: {weight}\n")
            f.write(f"c1: {c1}\n")
            f.write(f"c2: {c2}\n")
            f.write(f"func_num: {func_num}\n")
            f.write(f"num_dims: {num_dims}\n")
            f.write(f"lower_bound: {lower_bound}\n")
            f.write(f"upper_bound: {upper_bound}\n")
            f.write(f"iterations: {iterations}\n")

        return

    def log(
        self,
        iteration: int,
        particles: list[Particle],
        global_best_position: np.ndarray,
        global_best_value: np.float64,
    ) -> None:
        """Log the optimization progress.

        Args:
            iteration (int): Current iteration number
            particles (list[Particle]): Particles
            global_best_position (np.ndarray): Current global best position
            global_best_value (np.float64): Current global best value
        """

        # Add the current positions and values of the particles to the log data.
        self.log_data["iteration"][iteration] = iteration
        self.log_data["positions"][iteration] = np.array(
            [particle.position for particle in particles]
        )
        self.log_data["values"][iteration] = np.array(
            [particle.value for particle in particles]
        )

        # Add the global best position and value to the log data.
        self.global_best_log_data["iteration"][iteration] = iteration
        self.global_best_log_data["position"][iteration] = global_best_position
        self.global_best_log_data["value"][iteration] = global_best_value

        # Save the log data to the log files.
        np.savez_compressed(
            file=f"{self.path}/log_data.npz",
            iteration=self.log_data["iteration"],
            positions=self.log_data["positions"],
            values=self.log_data["values"],
        )

        # Plot the optimization progress using global best log data.
        plt.plot(
            self.global_best_log_data["iteration"], self.global_best_log_data["value"]
        )
        plt.title("Optimization Progress")
        plt.xlabel("Iteration")
        plt.ylabel("Global Best Value")
        plt.grid()
        plt.savefig(f"{self.path}/optimization_progress.svg", format="svg")
        plt.savefig(
            f"{self.path}/optimization_progress_transparent.svg",
            format="svg",
            transparent=True,
        )
        plt.close()

        return

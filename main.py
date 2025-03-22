import argparse

import numpy as np
from cec2013single.cec2013 import Benchmark
from tqdm import tqdm

from particle import Particle
from utils import *


def main() -> None:
    """ main of PSO program.

    Args:
        seed (int): Seed of random.
        iteration (int): Iteration.
        n (int): Number of particles.
        limit_init_speed (float): Limit of initial speed.
        weight (float): weight.
        c1 (float): c1.
        c2 (float): c2.
        function (int): Number of function.
        dimension (int): Dimension.
    """

    # Define arguments.
    parser = argparse.ArgumentParser(
        prog="Particle Swarm Optimization", description="Python program of PSO.")
    parser.add_argument("--seed", action="store",
                        type=int, help="seed")
    parser.add_argument("--iteration", "--iter", "-i", action="store",
                        type=int, default=1000, help="iteration (default: 1000)")
    parser.add_argument("--n", action="store", default=30,
                        type=int, help="number of particles (default: 1e-3)")
    parser.add_argument("--limit-init-speed", action="store", default=10,
                        type=float, help="limit of initial speed (default: 10)")
    parser.add_argument("--weight", "-w", action="store",
                        default=0.729, type=float, help="weight (default: 0.729)")
    parser.add_argument("--c1", action="store", default=1.4955,
                        type=float, help="c1 (default: 1.4955)")
    parser.add_argument("--c2", action="store", default=1.4955,
                        type=float, help="c2 (default: 1.4955)")
    parser.add_argument("--function", "--func", "-f", action="store",
                        type=int, required=True, help="function number")
    parser.add_argument("--dimension", "--dim", "-d", action="store",
                        default=10, type=int, help="function dimension (default: 10)")
    parser.add_argument("--save", action="store",
                        default=None, type=str, help="save directory")
    args = parser.parse_args()

    # Check whether log directory exits.
    check_log_dir()

    # Set seed.
    seed = args.seed
    if not seed:
        np.random.seed(seed)

    # Init.
    dim = args.dimension
    function_number = args.function
    benchmark = Benchmark()
    benchmark_function = benchmark.get_function(function_number)
    info = benchmark.get_info(function_number, dim)
    upper = info["upper"]
    lower = info["lower"]
    particles = [Particle(dim, upper, lower, args.limit_init_speed,
                          args.weight, args.c1, args.c2) for _ in range(args.n)]
    global_best = float("inf")
    global_best_position = np.full(dim, float("inf"))
    history = {
        "global_best": [],
        "global_best_position": []
    }
    evaluate(particles, benchmark_function)
    update_personal_best(particles)
    global_best, global_best_position = update_global_best(
        particles, global_best, global_best_position, history)

    # Search.
    for _ in tqdm(range(args.iteration)):
        update_speed(particles, global_best_position)
        update_position(particles)
        evaluate(particles, benchmark_function)
        update_personal_best(particles)
        global_best, global_best_position = update_global_best(
            particles, global_best, global_best_position, history)

    # log.
    particles_history = [particle.get_history() for particle in particles]
    log(vars(args), args.save, global_best, history, particles_history)
    return


def evaluate(particles, function) -> None:
    for particle in particles:
        particle.update_fitness(function(particle.get_position()))
    return


def update_position(particles) -> None:
    for particle in particles:
        particle.update_position()
    return


def update_speed(particles, global_best_position) -> None:
    for particle in particles:
        particle.update_speed(global_best_position)
    return


def update_personal_best(particles) -> None:
    for particle in particles:
        particle.update_personal_best()
    return


def update_global_best(particles, global_best, global_best_position, history) -> float:
    for particle in particles:
        personal_best = particle.get_personal_best()
        if personal_best < global_best:
            global_best = personal_best
            global_best_position = particle.get_personal_best_position()
    history["global_best"].append(global_best)
    history["global_best_position"].append(global_best_position)
    return global_best, global_best_position


if __name__ == "__main__":
    main()

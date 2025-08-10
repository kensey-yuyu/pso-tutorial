import argparse

from cec2013single.cec2013 import Benchmark
from pso import PSO
from utils import Logger, set_seed


def main() -> None:
    """Main function for PSO-Tutorial.
    """

    # Define arguments.
    parser = argparse.ArgumentParser(
        prog="PSO-Tutorial", description="Tutorial for Particle Swarm Optimization(PSO). These programs are  written by Python 3. If you want the PSO which run fast, you need to find others.")
    parser.add_argument("--seed", action="store",
                        type=int, help="seed value for radom number generation")
    parser.add_argument("--function", "--func", "-f", action="store",
                        type=int, required=True, help="function number of cec2013 single objective optimization")
    parser.add_argument("--dimension", "--dim", "-d", action="store",
                        default=10, type=int, help="number of dimension (default: 10)")
    parser.add_argument("--n", "-n", action="store", default=30,
                        type=int, help="number of particles (default: 30)")
    parser.add_argument("--iterations", "--iter", "-i", action="store",
                        type=int, default=1000, help="number of iteration (default: 1000)")
    parser.add_argument("--limit-init-velocity", action="store", default=0,
                        type=float, help="limit of initial velocity (default: 0)")
    parser.add_argument("--weight", action="store",
                        default=0.729, type=float, help="weight hyperparameter for update particle positions (default: 0.729)")
    parser.add_argument("--c1", action="store", default=1.4955,
                        type=float, help="c1 hyperparameter for update particle positions (default: 1.4955)")
    parser.add_argument("--c2", action="store", default=1.4955,
                        type=float, help="c2 hyperparameter for update particle positions (default: 1.4955)")
    parser.add_argument("--save-name", action="store",
                        default=None, type=str, help="directory name to save results")
    args = parser.parse_args()

    # Initialize logger.
    logger = Logger(args.save_name)

    # Set seed.
    seed: int = args.seed
    if not seed:
        set_seed(seed)

    # Load benchmark function from cec2013single written by dmolina (https://github.com/dmolina/cec2013single).
    function_number: int = args.function
    dim: int = args.dimension
    benchmark = Benchmark()
    benchmark_function = benchmark.get_function(
        function_number)
    info = benchmark.get_info(function_number, dim)
    upper: int = info["upper"]
    lower: int = info["lower"]

    # Initialize PSO.
    pso = PSO(args.n, args.iterations, dim, float(upper), float(lower), args.limit_init_velocity,
              args.weight, args.c1, args.c2, benchmark_function)

    # Explore optimum.
    pso.optimize()

    # Save log.
    logger.save(pso.get_history())

    return


if __name__ == "__main__":
    main()

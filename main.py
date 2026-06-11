import argparse

from pso.pso import PSO


def main(
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
    """Main function for the Particle Swarm Optimization.

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

    print("Hello from pso-tutorial!\n")

    # Shows the parameters defined as arguments.
    print(f"num_particles: {num_particles}")
    print(f"weight: {weight}")
    print(f"c1: {c1}")
    print(f"c2: {c2}")
    print(f"func_num: {func_num}")
    print(f"num_dims: {num_dims}")
    print(f"lower_bound: {lower_bound}")
    print(f"upper_bound: {upper_bound}")
    print(f"iterations: {iterations}\n")

    # Particle Swarm Optimization (PSO) is initialized.
    pso = PSO(
        num_particles=num_particles,
        weight=weight,
        c1=c1,
        c2=c2,
        func_num=func_num,
        num_dims=num_dims,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        iterations=iterations,
    )
    pso.optimize()

    return


if __name__ == "__main__":
    # Parses the command-line arguments.
    parser = argparse.ArgumentParser(description="Run the PSO tutorial.")
    parser.add_argument("--num-particles", type=int, help="Number of particles")
    parser.add_argument("--weight", type=float, help="Inertia weight")
    parser.add_argument("--c1", type=float, help="Cognitive parameter")
    parser.add_argument("--c2", type=float, help="Social parameter")
    parser.add_argument(
        "--func-num",
        type=int,
        help="Function number to optimize 1: Sphere, 2: Rastrigin",
    )
    parser.add_argument("--num-dims", type=int, help="Number of dimensions")
    parser.add_argument(
        "--lower-bound",
        type=float,
        help="Lower bound of the search space",
    )
    parser.add_argument(
        "--upper-bound",
        type=float,
        help="Upper bound of the search space",
    )
    parser.add_argument("--iterations", type=int, help="Number of iterations")
    args = parser.parse_args()
    args_dict = vars(args)

    main(**args_dict)

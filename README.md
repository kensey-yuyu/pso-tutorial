# PSO-Tutorial

This is a tutorial for Particle Swarm Optimization (PSO).
The code in this repository are simple and easy to understand.
As a result, the performance is not optimized.

## How to run

This project us as the package manager.
You can configure the optimization parameters using command-line options.
Run the PSO using the command below:

```bash
uv run main.py --num-particles NUM_PARTICLES \
               --weight WEIGHT \
               --c1 C1 \
               --c2 C2 \
               --func-num FUNC_NUM \
               --num-dims NUM_DIMS \
               --lower-bound LOWER_BOUND \
               --upper-bound UPPER_BOUND \
               --iterations ITERATIONS
```

Use the `--func-num` option to specify the objective function to optimize.
Run with `--help` for details on available functions.
Currently, the **sphere function** and **rastrigin function** are implemented as examples.

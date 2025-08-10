# PSO-Tutorial

Particle Swarm Optimization (PSO) is one of the swarm intelligence methods.
This repository is written by Python and can be used to search the CEC2013 benchmarks.
It is coded for beginners, so the programs are probably easy to read.
Therefore, its execution speed is not very fast.

## Setup

### Environments

- Cython: 3.0.12
- matplotlib: 3.10.0
- numpy: 2.2.3
- cec2013single: 0.1
- setuptools: 75.8.1
- wheel: 0.45.1
- tqdm: 4.67.1

### bash for setup environments

```bash
cd PSO
bash setup.sh
```

## Run

The number of particles, iterations and dimensions etc. are defined as arguments.
The hyperparameters $w$, $c1$, $c2$ are also defined in the same way.
Please check argument help.

```bash
python3 main.py -f (any_number)
```

## Reference

- cec2013single: <https://github.com/dmolina/cec2013single>
- Problem Definitions and Evaluation Criteria for the CEC 2013 Special Session on Real-Parameter Optimization: <https://www.al-roomi.org/multimedia/CEC_Database/CEC2013/RealParameterOptimization/CEC2013_RealParameterOptimization_TechnicalReport.pdf>

## TODO

- Replace existing evaluation function codes with manually written implementations

import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt


def check_log_dir():
    # Check directory.
    if not os.path.isdir("./log"):
        print("No log directory. Create log directory...")
        try:
            os.mkdir("./log")
        except OSError:
            exit("Error: Could not make log directory.")
    return


def log(args, path, global_best, global_history, particles_history):
    # Try to create directory.
    if not path:
        path = "./log/" + format(datetime.now(), "%Y-%m-%d_%H:%M:%S")
    else:
        path = "./log/" + path

    try:
        os.mkdir(path)
    except OSError:
        exit("Error: Could not make log directory.")

    # Write log.
    with open(f"{path}/log.txt", "w", encoding="utf-8") as file:
        for key, value in args.items():
            file.write(str(f"{key}: {value}\n"))

    # Save history.
    history = {
        "global": global_history,
        "particles": particles_history
    }
    with open(f"{path}/history.pickle", mode="wb") as file:
        pickle.dump(history, file)

    # Plot graph of global best.
    plot_global_best(global_best, global_history, path)
    return


def plot_global_best(global_best, history, path):
    # Plot graph of global best.
    plt.figure()
    plt.title(f"Iteration and Global best\n Global best: {global_best}")
    plt.xlabel("Iteration")
    plt.ylabel("Global best")
    plt.plot(history["global_best"])
    plt.savefig(f"{path}/result.png")
    plt.close()

    # Plot log-scale graph of global best.
    plt.figure()
    plt.yscale("log")
    plt.title(f"Iteration and Global best\n Global best: {global_best}")
    plt.xlabel("Iteration")
    plt.ylabel("Global best")
    plt.plot(history["global_best"])
    plt.savefig(f"{path}/result_log-scale.png")
    plt.close()
    return

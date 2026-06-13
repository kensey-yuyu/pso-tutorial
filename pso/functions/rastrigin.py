import numpy as np


def rastrigin(x: np.ndarray) -> np.float64:
    """Rastrigin function.

    Args:
        x (np.ndarray): Input array

    Returns:
        np.float64: Output value
    """

    A: int = 10
    return A * x.size + np.sum(np.square(x) - A * np.cos(2 * np.pi * x))

import numpy as np


def sphere(x: np.ndarray) -> np.float64:
    """Sphere function.

    Args:
        x (np.ndarray): Input array

    Returns:
        np.float64: Output value
    """

    return np.sum(np.square(x))

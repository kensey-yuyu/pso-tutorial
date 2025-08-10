import random

import numpy as np


def set_seed(seed: int) -> None:
    """set seed for random number generation.

    Args:
        seed (int): Seed value.
    """

    random.seed(seed)
    np.random.seed(seed)
    return

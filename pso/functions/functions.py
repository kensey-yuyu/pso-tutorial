from collections.abc import Callable

import numpy as np

from pso.functions.rastrigin import rastrigin
from pso.functions.sphere import sphere


class Functions:
    def __init__(self, func_num: int, num_dims: int) -> None:
        """Initialize the functions.

        Args:
            func_num (int): Function number to optimize
            num_dims (int): Number of dimensions
        """

        self.func_num = func_num
        self.num_dims = num_dims

        return

    def get_func(self) -> Callable[[np.ndarray], np.float64]:
        """Get the function to optimize.

        Raises:
            ValueError: Invalid function number

        Returns:
            Callable[[np.ndarray], np.float64]: The function to optimize
        """

        if self.func_num == 1:
            return sphere
        elif self.func_num == 2:
            return rastrigin
        else:
            raise ValueError(f"Invalid function number: {self.func_num}")

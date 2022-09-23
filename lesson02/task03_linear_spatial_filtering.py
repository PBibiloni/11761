from typing import Tuple

import numpy as np


def kernel_mean_filter(size: Tuple[int, int]) -> np.ndarray:
    """Returns a kernel the of given size for the mean filter."""
    # YOUR CODE HERE
    # ...
    return np.zeros(shape=size, dtype=np.float32)


def kernel_gaussian_filter(size: Tuple[int, int], sigma: float) -> np.ndarray:
    """Returns a kernel of the given size for the Gaussian filter."""
    # YOUR CODE HERE
    # ...
    return np.zeros(shape=size, dtype=np.float32)


def kernel_horizontal_derivative_gaussian_filter(size: Tuple[int, int], sigma: float) -> np.ndarray:
    """Returns a kernel of the given size for the Gaussian filter, using first order central difference. """
    # YOUR CODE HERE
    # ...
    return np.zeros(shape=size, dtype=np.float32)


def kernel_vertical_derivative_gaussian_filter(size: Tuple[int, int], sigma: float) -> np.ndarray:
    """Returns a kernel of the given size for the Gaussian filter, using first order central difference. """
    # YOUR CODE HERE
    # ...
    return np.zeros(shape=size, dtype=np.float32)


def kernel_LoG_filter(size: Tuple[int, int], sigma: float) -> np.ndarray:
    """Returns a kernel of the given size for the Laplacian of Gaussian filter."""
    # YOUR CODE HERE
    # ...
    return np.zeros(shape=size, dtype=np.float32)
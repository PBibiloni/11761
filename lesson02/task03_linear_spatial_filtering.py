from typing import Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import sample_filepath


def kernel_squared_mean_filter(size: Tuple[int, int]) -> np.ndarray:
    """Returns a kernel the of given size for the mean filter."""
    # YOUR CODE HERE:
    #   See `np.ones(...)`.
    #   ...
    return np.ones(shape=size, dtype=np.float32) / np.prod(size)


def kernel_gaussian_filter(size: Tuple[int, int], sigma: float) -> np.ndarray:
    """Returns a kernel of the given size for the Gaussian filter."""
    # YOUR CODE HERE:
    #   See `np.exp(...)`.
    #   ...
    kernel = np.zeros(shape=size, dtype=np.float32)
    for i in range(size[0]):
        for j in range(size[1]):
            kernel[i, j] = np.exp(-((i-size[0]//2)**2 + (j-size[1]//2)**2)/(2*sigma**2))
    kernel = kernel/np.sum(kernel)
    return kernel


def kernel_sharpening(kernel_smoothing: np.ndarray, alpha: float) -> np.ndarray:
    """Returns a kernel for sharpening the image."""
    # YOUR CODE HERE:
    #   See `np.zeros(...)` and `np.zeros_like(...)`.
    #   ...
    sz = kernel_smoothing.shape
    kernel_impulse = np.zeros_like(kernel_smoothing)
    kernel_impulse[sz[0]//2, sz[1]//2] = 1
    kernel_detail = kernel_impulse - kernel_smoothing
    kernel_sharpening = kernel_impulse + alpha * kernel_detail
    return kernel_sharpening


def kernel_horizontal_derivative() -> np.ndarray:
    """Returns a 3x1 kernel for the horizontal derivative using first order central difference coefficients. """
    # YOUR CODE HERE:
    #   ...
    return np.array([[1/2, 0, -1/2]])


def kernel_vertical_derivative() -> np.ndarray:
    """Returns a 1x3 kernel for the vertical derivative using first order central difference coefficients. """
    # YOUR CODE HERE:
    #   See `np.transpose(...)`.
    #   ...
    return kernel_horizontal_derivative().transpose()


def kernel_sobel_horizontal() -> np.ndarray:
    """Returns the sobel operator for horizontal derivatives. """
    # YOUR CODE HERE:
    #   ...
    sobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)/8
    return sobel


def kernel_sobel_vertical() -> np.ndarray:
    """Returns the sobel operator for vertical derivatives. """
    # YOUR CODE HERE:
    #   See `np.transpose(...)`.
    #   ...
    return kernel_sobel_horizontal().transpose()


def kernel_LoG_filter() -> np.ndarray:
    """Returns a 3x3 kernel for the Laplacian of Gaussian filter."""
    # YOUR CODE HERE:
    #   ...
    log = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
    return log


if __name__ == '__main__':
    # Show effect on subset of boat image.
    img = cv2.imread(sample_filepath('boat.tiff'), cv2.IMREAD_GRAYSCALE)  # Read the image.
    img = img.astype('float32')     # Convert to float32 to avoid overflow and rounding errors
    img = img[150:200, 150:200]     # Select a small window

    kernels = {
        'kernel_squared_mean_filter': kernel_squared_mean_filter(size=(3, 3)),
        'kernel_gaussian_filter': kernel_gaussian_filter(size=(3, 3), sigma=10.0),
        'kernel_sharpening': kernel_sharpening(kernel_squared_mean_filter(size=(3, 3)), alpha=2),
        'kernel_horizontal_derivative': kernel_horizontal_derivative(),
        'kernel_vertical_derivative': kernel_vertical_derivative(),
        'kernel_sobel_horizontal': kernel_sobel_horizontal(),
        'kernel_sobel_vertical': kernel_sobel_vertical(),
        'kernel_LoG_filter': kernel_LoG_filter(),
    }
    kernels = {k: v for k, v in kernels.items() if v is not None}  # Remove None values.

    for name, kernel in kernels.items():
        output = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)  # ddepth=-1 means same as input
        # Visualize images
        fig, axs = plt.subplots(1, 2)
        axs[0].set_title(name)
        im = axs[0].imshow(kernel, cmap='gray')
        plt.colorbar(im, ax=axs[0])
        axs[1].imshow(output, cmap='gray')
        # Display figure
        plt.show()

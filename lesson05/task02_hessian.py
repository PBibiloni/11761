import typing as tp

import cv2
import numpy as np
from matplotlib import pyplot as plt, colors

from utils import sample_filepath


def hessian_matrix(img_grayscale: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns the hessian matrix of the image."""
    # YOUR CODE HERE:
    #   Use `cv2.Sobel(..., dx= , dy= , ...)` to compute second-order derivatives
    #   ...


def hessian_eigenvalues(img_grayscale: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Returns the eigenvalues of the hessian matrix of the image."""
    # YOUR CODE HERE:
    #   Remember that eigenvalues are solutions of `x^2 - trace * x + det = 0`
    #   ...


def cylinders(img_grayscale: np.ndarray) -> np.ndarray:
    """Returns the pixels of the image that correspond to dark cylinder-like structure."""
    # YOUR CODE HERE:
    #   Use the eigenvalues of the Hessian to characterize thin structures.
    #   ...


if __name__ == "__main__":
    img = cv2.imread(sample_filepath('Retinal_DRIVE21_original.tif'), cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Visualize `hessian_matrix(...)`
    hessian_dxdx, hessian_dxdy, hessian_dydx, hessian_dydy = hessian_matrix(img_gray)
    min_value = np.min([hessian_dxdx, hessian_dxdy, hessian_dydx, hessian_dydy])
    max_value = np.max([hessian_dxdx, hessian_dxdy, hessian_dydx, hessian_dydy])
    colormap_normalizer = colors.SymLogNorm(linthresh=0.1, vmin=min_value, vmax=max_value) #For better visualization
    _, axs = plt.subplots(2, 3)
    axs[0, 0].set_title('Original')
    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title('Grayscale')
    axs[1, 0].imshow(img_gray, cmap='gray')
    axs[0, 1].set_title('hessian_dxdx')
    axs[0, 1].imshow(hessian_dxdx, cmap='gray', norm=colormap_normalizer)
    axs[0, 2].set_title('hessian_dxdy')
    axs[0, 2].imshow(hessian_dxdy, cmap='gray', norm=colormap_normalizer)
    axs[1, 1].set_title('hessian_dydx')
    axs[1, 1].imshow(hessian_dydx, cmap='gray', norm=colormap_normalizer)
    axs[1, 2].set_title('hessian_dydy')
    axs[1, 2].imshow(hessian_dydy, cmap='gray', norm=colormap_normalizer)
    plt.show()

    # Visualize `hessian_eigenvalues(...)`
    eigenvalue_1, eigenvalue_2 = hessian_eigenvalues(img_gray)
    hessian_det = eigenvalue_1 * eigenvalue_2
    hessian_trace = eigenvalue_1 + eigenvalue_2
    colormap_normalizer = colors.SymLogNorm(linthresh=0.1, vmin=np.min([eigenvalue_1, eigenvalue_2]), vmax=np.max([eigenvalue_1, eigenvalue_2])) # For better visualization
    _, axs = plt.subplots(2, 3)
    axs[0, 0].set_title('Original')
    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title('Grayscale')
    axs[1, 0].imshow(img_gray, cmap='gray')
    axs[0, 1].set_title('eigenvalue_1')
    axs[0, 1].imshow(eigenvalue_1, cmap='gray', norm=colormap_normalizer)
    axs[0, 2].set_title('eigenvalue_2')
    axs[0, 2].imshow(eigenvalue_2, cmap='gray', norm=colormap_normalizer)
    axs[1, 1].set_title('det')
    axs[1, 1].imshow(hessian_det, cmap='gray', norm=colors.SymLogNorm(linthresh=0.1))
    axs[1, 2].set_title('trace')
    axs[1, 2].imshow(hessian_trace, cmap='gray', norm=colors.SymLogNorm(linthresh=0.1))
    plt.show()

    # Visualize `find_cylinders(...)`
    img_cylinders = cylinders(img_gray)
    img_with_cylinders = img.copy()
    img_with_cylinders[img_cylinders == 255, :] = [0, 255, 0]
    _, axs = plt.subplots(1, 2)
    # Show one image per subplot
    axs[0].set_title('Original')
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Cylinders')
    axs[1].imshow(cv2.cvtColor(img_with_cylinders, cv2.COLOR_BGR2RGB))
    plt.show()

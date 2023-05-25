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
    hessian_dxdx = cv2.Sobel(img_grayscale, cv2.CV_32F, 2, 0, ksize=3)
    hessian_dxdy = cv2.Sobel(img_grayscale, cv2.CV_32F, 1, 1, ksize=3)
    hessian_dydx = hessian_dxdy
    hessian_dydy = cv2.Sobel(img_grayscale, cv2.CV_32F, 0, 2, ksize=3)
    return hessian_dxdx, hessian_dxdy, hessian_dydx, hessian_dydy


def hessian_eigenvalues(img_grayscale: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Returns the eigenvalues of the hessian matrix of the image."""
    # YOUR CODE HERE:
    #   Remember that eigenvalues are solutions of `x^2 - trace * x + det = 0`
    #   ...
    hessian_dxdx, hessian_dxdy, hessian_dydx, hessian_dydy = hessian_matrix(img_grayscale)
    hessian_det = hessian_dxdx * hessian_dydy - hessian_dxdy * hessian_dydx
    hessian_trace = hessian_dxdx + hessian_dydy
    # Solve `x^2 - trace * x + det = 0`
    hessian_eigenvalue_1 = 0.5 * (hessian_trace + np.sqrt(hessian_trace**2 - 4 * hessian_det))
    hessian_eigenvalue_2 = 0.5 * (hessian_trace - np.sqrt(hessian_trace**2 - 4 * hessian_det))
    # Order eigenvalues (for every pixel)
    hessian_eigenvalue_1, hessian_eigenvalue_2 = np.max([hessian_eigenvalue_1, hessian_eigenvalue_2], axis=0), np.min([hessian_eigenvalue_1, hessian_eigenvalue_2], axis=0)
    return hessian_eigenvalue_1, hessian_eigenvalue_2


def cylinders(img_grayscale: np.ndarray) -> np.ndarray:
    """Returns the pixels of the image that correspond to dark cylinder-like structure."""
    cylinders = np.zeros_like(img_grayscale)
    # YOUR CODE HERE:
    #   Remember that eigenvalues are solutions of `x^2 - trace * x + det = 0`
    #   ...
    eig_1, eig_2 = hessian_eigenvalues(img_grayscale)
    # Ensure they are ordered
    eig_1, eig_2 = np.max([eig_1, eig_2], axis=0), np.min([eig_1, eig_2], axis=0)
    # One eigenvalue should be large and positive, the other should be small in absolute value.
    threshold_high = 30
    threshold_flat_region = 10
    cylinders[(eig_1 > threshold_high) & (-threshold_flat_region < eig_2) & (eig_2 < threshold_flat_region)] = 255
    # Return value for visualization, whose pixels should be either 0 o 255.
    return cylinders


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

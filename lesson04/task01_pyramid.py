import typing as tp

import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import sample_filepath


def gaussian_pyramid(img: np.ndarray, levels: int) -> tp.List[np.ndarray]:
    """Returns a Gaussian pyramid of the image."""
    # YOUR CODE HERE:
    #   See `cv.pyrDown(...)`
    #   ...
    pyramid = []

    return pyramid


def laplacian_pyramid(img: np.ndarray, levels: int) -> tp.List[np.ndarray]:
    """Returns a Laplacian pyramid of the image."""
    # YOUR CODE HERE:
    #   See `cv.pyrDown(...)` and `cv.pyrUp(...)`
    #   ...
    l_pyramid = []

    return l_pyramid


def reconstruct_from_laplacian_pyramid(l_pyramid: tp.List[np.ndarray]) -> np.ndarray:
    """Reconstructs an image from its Laplacian pyramid."""
    # YOUR CODE HERE:
    #   See `cv.pyrUp(...)`, and start from the smallest layer.
    #   ...


def remove_finer_detail(img: np.ndarray) -> np.ndarray:
    """Removes the finer details of the image by applying a Laplacian pyramid."""
    # YOUR CODE HERE:
    #   Reuse `laplacian_pyramid(...)` and `reconstruct_from_laplacian_pyramid(...)`.
    #   ...


if __name__ == "__main__":
    # Show effect
    original_img = cv2.imread(sample_filepath('airplane.tiff'), cv2.IMREAD_GRAYSCALE)  # Read the image.
    original_img = original_img.astype('float32')  # Convert to float32 to avoid overflow and rounding errors

    pyramids = {
        'Gaussian': gaussian_pyramid(original_img, 4),
        'Laplacian': laplacian_pyramid(original_img, 4),
    }
    pyramids = {k: v for k, v in pyramids.items() if v is not None}  # Remove None values.

    for method_name, pyramid in pyramids.items():
        # Visualize images
        fig, axs = plt.subplots(1, 4)
        fig.suptitle(method_name)
        # Show one image per subplot
        for ax, subimage in zip(axs.flatten(), pyramid):
            ax.imshow(subimage, cmap='gray')
        # Display figure
        plt.show()

    # Show effect of removing coarse details
    if 'Laplacian' in pyramids:
        l_pyramid = pyramids['Laplacian']
        reconstructions = {
            'Original': original_img,
            'Recovered': reconstruct_from_laplacian_pyramid(l_pyramid),
            'Without coarse detail': remove_finer_detail(original_img),
        }
        reconstructions = {k: v for k, v in reconstructions.items() if v is not None}  # Remove None values.
        # Visualize images
        fig, axs = plt.subplots(1, 3)
        # Show one image per subplot
        for ax, (name, subimage) in zip(axs.flatten(), reconstructions.items()):
            ax.set_title(name)
            ax.imshow(subimage, cmap='gray')
        # Display figure
        plt.show()

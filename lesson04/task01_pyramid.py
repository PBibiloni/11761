import typing as tp

import cv2
import numpy as np
from matplotlib import pyplot as plt


def gaussian_pyramid(img: np.ndarray, levels: int) -> tp.List[np.ndarray]:
    """Returns a Gaussian pyramid of the image."""
    pyramid = [img]
    for _ in range(levels-1):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def laplacian_pyramid(img: np.ndarray, levels: int) -> tp.List[np.ndarray]:
    """Returns a Laplacian pyramid of the image."""
    g_pyramid = gaussian_pyramid(img, levels)
    l_pyramid = []
    for idx in range(levels-1):
        l_pyramid.append(g_pyramid[idx] - cv2.pyrUp(g_pyramid[idx+1]))
    l_pyramid.append(g_pyramid[-1])
    return l_pyramid


def reconstruct_from_laplacian_pyramid(l_pyramid: tp.List[np.ndarray]) -> np.ndarray:
    """Reconstructs an image from its Laplacian pyramid."""
    img = l_pyramid[-1]
    for level in range(len(l_pyramid)-2, -1, -1):
        img = cv2.pyrUp(img) + l_pyramid[level]
    return img


def remove_finer_detail(img: np.ndarray) -> np.ndarray:
    """Removes the finer details of the image by applying a Laplacian pyramid."""
    original_img = cv2.imread('../samples/airplane.tiff', cv2.IMREAD_GRAYSCALE).astype('float32')
    l_pyramid = laplacian_pyramid(original_img, levels=6)
    l_pyramid[0] = np.zeros_like(l_pyramid[0])
    l_pyramid[1] = np.zeros_like(l_pyramid[1])
    return reconstruct_from_laplacian_pyramid(l_pyramid)


if __name__ == "__main__":
    # Show effect
    original_img = cv2.imread('../samples/airplane.tiff', cv2.IMREAD_GRAYSCALE)  # Read the image.
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

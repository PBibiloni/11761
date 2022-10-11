import typing as tp

import cv2
import numpy as np
from matplotlib import pyplot as plt


def gaussian_pyramid(img: np.ndarray, levels: int) -> tp.List[np.ndarray]:
    """Returns a Gaussian pyramid of the image."""
    # Your code here: see `cv.pyrDown(...)`
    # ...
    pyramid = [img]
    for _ in range(levels):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def laplacian_pyramid(img: np.ndarray, levels: int) -> tp.List[np.ndarray]:
    """Returns a Laplacian pyramid of the image."""
    # Your code here: see `cv.pyrDown(...)` and `cv.pyrUp(...)`
    # ...
    g_pyramid = gaussian_pyramid(img, levels)
    l_pyramid = []
    for idx in range(levels-1):
        l_pyramid.append(g_pyramid[idx] - cv2.pyrUp(g_pyramid[idx+1]))
    l_pyramid.append(g_pyramid[-1])
    return l_pyramid


if __name__ == "__main__":
    # Show effect
    original_img = cv2.imread('../samples/airplane.tiff', cv2.IMREAD_GRAYSCALE)  # Read the image.
    original_img = original_img.astype('float32')  # Convert to float32 to avoid overflow and rounding errors

    results = {
        'Gaussian': gaussian_pyramid(original_img, 4),
        'Laplacian': laplacian_pyramid(original_img, 4),
    }
    results = {k: v for k, v in results.items() if v is not None}  # Remove None values.

    for method_name, pyramid in results.items():
        # Visualize images
        fig, axs = plt.subplots(1, 4)
        fig.suptitle(method_name)
        # Show one image per subplot
        for ax, subimage in zip(axs.flatten(), pyramid):
            ax.imshow(subimage, cmap='gray')
        # Display figure
        plt.show()

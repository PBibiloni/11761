import cv2
import numpy as np
import matplotlib.pyplot as plt

import lesson03.task01_binarize as task01
from utils import sample_filepath


def dilation(img: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Returns the dilation of the binary/grayscale image with the given structuring element."""
    # YOUR CODE HERE:
    #   See `cv2.dilate`
    #   ...


def erosion(img: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Returns the erosion of the binary/grayscale image with the given structuring element."""
    # YOUR CODE HERE:
    #   See `cv2.erode`
    #   ...


def opening(img: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Returns the opening of the binary/grayscale image with the given structuring element."""
    # YOUR CODE HERE:
    #   Reuse functions `dilation(...)` and `erosion(...)`.
    #   ...


def closing(img: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Returns the closing of the binary/grayscale image with the given structuring element."""
    # YOUR CODE HERE:
    #   Reuse functions `dilation(...)` and `erosion(...)`.
    #   ...


def morphological_gradient(img: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Returns the morphological gradient of the binary/grayscale image with the given structuring element."""
    # YOUR CODE HERE:
    #   ...


def morphological_skeleton(img: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Returns the morphological skeleton of the binary/grayscale image considering Lantuéjoul's method."""
    # YOUR CODE HERE:
    #   ...


if __name__ == "__main__":
    # Visualize morphological operations on binary and grayscale images
    binary_img = task01.binarize_by_otsu(cv2.imread(sample_filepath('dots.tiff'), cv2.IMREAD_GRAYSCALE))
    grayscale_img = cv2.imread(sample_filepath('mandril.tiff'), cv2.IMREAD_GRAYSCALE)

    se = np.ones((3, 3), dtype='uint8')    # 8-connectivity.

    for image in [binary_img, grayscale_img, grayscale_img[150:250, 150:250]]:
        results = {
            'erosion': erosion(image, se),
            'dilation': dilation(image, se),
            'original': image,
            'opening': opening(image, se),
            'closing': closing(image, se),
            'gradient': morphological_gradient(image, se),
            'skeleton': morphological_skeleton(image, se),
        }
        results = {k: v for k, v in results.items() if v is not None}  # Remove None values.

        # Visualize images
        fig, axs = plt.subplots(3, 3)
        # Remove default axis
        for ax in axs.flatten():
            ax.axis('off')
        # Show one image per subplot
        for ax, (title, subimage) in zip(axs.flatten(), results.items()):
            ax.set_title(title)
            ax.imshow(subimage, cmap='gray')
        # Display figure
        plt.show()

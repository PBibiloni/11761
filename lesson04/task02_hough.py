import typing as tp

import cv2
import numpy as np
from matplotlib import pyplot as plt


def hough_lines(img: np.ndarray):
    """Plots the location of the main lines of the image."""
    # Your code here: see `cv.HoughLines(...)` and the sample `logo.png`
    # ...


def hough_circles(img: np.ndarray) -> tp.List[np.ndarray]:
    """Plots the location of the main circles of the image."""
    # Your code here: see `cv.HoughCircles(...)` and the sample `dots.tiff`
    # ...


if __name__ == "__main__":
    visualizations = [
        ('Lines', hough_lines, '../samples/logo.png'),
        ('Circles', hough_circles, '../samples/dots.tiff'),
    ]

    for name, hough_function, path in visualizations:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(img, 100, 200)

        fig, axs = plt.subplots(1, 2)
        fig.suptitle(name)
        axs[0].set_title('Original')
        axs[0].imshow(img, cmap='gray')
        axs[1].set_title('Edges')
        axs[1].imshow(edges, cmap='gray')
        # Display figure
        plt.show()

        # Show effect
        res = hough_function(edges)

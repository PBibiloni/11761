import typing as tp

import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import sample_filepath


def hough_lines():
    """Draws and plots the location of the main lines of the image."""
    img = cv2.imread(sample_filepath('logo.png'), cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 200)
    # YOUR CODE HERE:
    #   Use `cv2.HoughLines(img_edges, ...)` with the `img_edges` as input.
    #   Use `cv2.line(img, ...)` to draw the lines on the original image `img`.
    #   ...


    # Visualize them
    fig, axs = plt.subplots(1, 3)
    fig.suptitle('Hough (lines)')
    axs[0].set_title('Grayscale')
    axs[0].imshow(img_gray, cmap='gray')
    axs[1].set_title('Edges (dilated)')
    axs[1].imshow(cv2.dilate(img_edges, np.ones((5,5))), cmap='gray')
    axs[2].set_title('Img (with lines)')
    axs[2].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap='gray')
    plt.show()


def hough_circles():
    """Draws and plots the location of the main circles of the image."""
    img = cv2.imread(sample_filepath('dots.tiff'), cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 200)
    # YOUR CODE HERE:
    #   See `cv.HoughCircles(img_edges, ...)` with the `img_edges` as input.
    #   Use `cv2.circle(img, ...)` to draw the circles on the original image `img`.
    #   ...


    # Visualize them
    fig, axs = plt.subplots(1, 3)
    fig.suptitle('Hough (lines)')
    axs[0].set_title('Grayscale')
    axs[0].imshow(img_gray, cmap='gray')
    axs[1].set_title('Edges (dilated)')
    axs[1].imshow(cv2.dilate(img_edges, np.ones((5,5))), cmap='gray')
    axs[2].set_title('Img (with lines)')
    axs[2].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap='gray')
    plt.show()


if __name__ == "__main__":
    hough_lines()
    hough_circles()

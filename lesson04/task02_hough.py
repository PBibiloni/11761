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
    lines = cv2.HoughLines(img_edges, rho=1, theta=np.pi/60, threshold=150)
    if lines is not None:
        for line in lines:
            r, theta = line[0]
            if theta > np.pi/4:
                # Depending on the angle, parameterize the line as x = r * cos(theta) or y = r * sin(theta)
                line_start = (0, int(r/np.sin(theta)))
                line_end = (img.shape[0], int((r-img.shape[0]*np.cos(theta))/np.sin(theta)))
                cv2.line(img, line_start, line_end, (0, 255, 0), 3, cv2.LINE_AA)
            else:
                line_start = (int(r/np.cos(theta)), 0)
                line_end = (int((r-img.shape[1]*np.sin(theta))/np.cos(theta)), img.shape[1])
                cv2.line(img, line_start, line_end, (0, 0, 255), 3, cv2.LINE_AA)

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
    circles = cv2.HoughCircles(img_edges, method=cv2.HOUGH_GRADIENT, dp=1,
                               minDist=img_edges.shape[0] // 16,
                               param2=20, minRadius=1, maxRadius=img_edges.shape[0] // 8)
    if circles is not None:
        for circle in circles[0]:
            center = (int(circle[0]), int(circle[1]))
            radius = int(circle[2])
            cv2.circle(img, center, 1, color=(0, 255, 0), thickness=3)          # Plot the center
            cv2.circle(img, center, radius, color=(0, 0, 255), thickness=3)   # And the circumference

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

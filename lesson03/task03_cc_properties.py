import cv2
import numpy as np


def largest_object(binary_img: np.ndarray) -> np.ndarray:
    """Returns a binary image with only the largest connected component."""
    # YOUR CODE HERE: see `cv2.ConnectedComponentsTypes`
    # ...
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img.astype('uint8'), connectivity=4)
    largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    # TODO
    return labels == largest_label


def most_centered_object(binary_img: np.ndarray) -> np.ndarray:
    """Returns a binary image with only the most centered connected component."""
    # YOUR CODE HERE: see `cv2.ConnectedComponentsTypes`
    # ...
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img.astype('uint8'), connectivity=4)
    # TODO
    largest_label = np.argmin(np.linalg.norm(centroids[1:], axis=1)) + 1
    return labels == largest_label


def most_rectangular_object(binary_img: np.ndarray) -> np.ndarray:
    """Returns a binary image with only the most rectangular connected component."""
    # YOUR CODE HERE: see `cv2.ConnectedComponentsTypes`
    # ...
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img.astype('uint8'), connectivity=4)
    # TODO
    largest_label = np.argmax(stats[1:, cv2.CC_STAT_WIDTH] / stats[1:, cv2.CC_STAT_HEIGHT]) + 1
    return labels == largest_label


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img = cv2.imread("../samples/tank.tiff", cv2.IMREAD_GRAYSCALE)
    binary_img = img < 127
    largest_object_img = largest_object(binary_img)
    plt.imshow(largest_object_img, cmap="gray")
    plt.show()

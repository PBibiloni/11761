import cv2
import numpy as np


def dilation(binary_img: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Returns the dilation of the binary image with the given structuring element."""
    # YOUR CODE HERE: see `cv2.dilate`
    # ...
    return cv2.dilate(binary_img.astype('uint8'), structuring_element.astype('uint8'))


def erosion(binary_img: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Returns the erosion of the binary image with the given structuring element."""
    # YOUR CODE HERE: see `cv2.erode`
    # ...
    return cv2.erode(binary_img.astype('uint8'), structuring_element.astype('uint8'))


def opening(binary_img: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Returns the opening of the binary image with the given structuring element."""
    # YOUR CODE HERE: see `cv2.dilate` and `cv2.erode`
    # ...
    return dilation(erosion(binary_img, structuring_element), structuring_element)


def closing(binary_img: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Returns the closing of the binary image with the given structuring element."""
    # YOUR CODE HERE: see `cv2.dilate` and `cv2.erode`
    # ...
    return erosion(dilation(binary_img, structuring_element), structuring_element)


def morphological_gradient(binary_img: np.ndarray) -> np.ndarray:
    """Returns the morphological gradient of the binary image with the given structuring element."""
    structuring_element = np.ones((3, 3), dtype='uint8')    # 8-connectivity.
    # YOUR CODE HERE: see `cv2.morphologyEx`
    # ...
    # TODO
    return cv2.morphologyEx(binary_img.astype('uint8'), cv2.MORPH_GRADIENT, structuring_element.astype('uint8'))


def morphological_skeleton(binary_img: np.ndarray) -> np.ndarray:
    """Returns the morphological skeleton of the binary image considering the Lantuéjoul's formula."""
    structuring_element = np.ones((3, 3), dtype='uint8')    # 8-connectivity.
    # YOUR CODE HERE: see `cv2.morphologyEx`
    # ...
    # TODO
    return cv2.morphologyEx(binary_img.astype('uint8'), cv2.MORPH_SKELETON, structuring_element.astype('uint8'))


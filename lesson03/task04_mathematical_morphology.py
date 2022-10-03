import cv2
import numpy as np
import matplotlib.pyplot as plt

import lesson03.task01_binarize as task01


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


def morphological_gradient(binary_img: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Returns the morphological gradient of the binary image with the given structuring element."""
    # YOUR CODE HERE: see `cv2.morphologyEx`
    # ...
    return dilation(binary_img, structuring_element) - erosion(binary_img, structuring_element)


def morphological_skeleton(binary_img: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Returns the morphological skeleton of the binary image considering Lantuéjoul's method."""
    # YOUR CODE HERE: see `cv2.morphologyEx`
    # ...
    # Iteratively erode the image (until there are no more pixels)
    eroded_imgs = []
    current_img = binary_img
    while np.any(current_img):
        eroded_imgs.append(current_img)
        current_img = erosion(current_img, structuring_element)

    skeleton = np.zeros_like(binary_img)
    for eroded_img in eroded_imgs:
        skeleton += eroded_img - opening(eroded_img, structuring_element)
    return skeleton


if __name__ == "__main__":
    # Visualize morphological operations on binary and grayscale images
    binary_img = task01.binarize_by_Otsu(cv2.imread("../samples/dots.tiff", cv2.IMREAD_GRAYSCALE))
    grayscale_img = cv2.imread("../samples/mandril.tiff", cv2.IMREAD_GRAYSCALE)

    structuring_element = np.ones((3, 3), dtype='uint8')    # 8-connectivity.

    for img in [binary_img, grayscale_img]:
        results = {
            'erosion': erosion(img, structuring_element),
            'dilation': dilation(img, structuring_element),
            'original': img,
            'opening': opening(img, structuring_element),
            'closing': closing(img, structuring_element),
            'gradient': morphological_gradient(img, structuring_element),
            'skeleton': morphological_skeleton(img, structuring_element),
        }
        results = {k: v for k, v in results.items() if v is not None}  # Remove None values.

        # Visualize images
        fig, axs = plt.subplots(3, 3)
        # Remove default axis
        for ax in axs.flatten():
            ax.axis('off')
        # Show one image per subplot
        for ax, (title, img) in zip(axs.flatten(), results.items()):
            ax.set_title(title)
            ax.imshow(img, cmap='gray')
        # Display figure
        plt.show()

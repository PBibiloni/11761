import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import sample_filepath


def binarize_by_thresholding(img: np.ndarray, threshold: float) -> np.ndarray:
    """Returns a binary version of the image by applying a thresholding operation."""
    # YOUR CODE HERE
    #   ...
    return (img >= threshold)*255


def binarize_by_otsu(img: np.ndarray) -> np.ndarray:
    """Returns a binary version of the image by applying a thresholding operation."""
    otsu_threshold = 0
    lowest_criteria = np.inf
    for threshold in range(255):
        # YOUR CODE HERE:
        #   Assume that img ranges from 0 to 255.
        #   ...
        thresholded_im = img >= threshold
        # compute weights
        weight1 = np.sum(thresholded_im) / img.size
        weight0 = 1 - weight1

        # if one the classes is empty, that threshold will not be considered
        if weight1 != 0 and weight0 != 0:
            # compute criteria, based on variance of these classes
            var0 = np.var(img[thresholded_im == 0])
            var1 = np.var(img[thresholded_im == 1])
            otsu_criteria = weight0 * var0 + weight1 * var1

            if otsu_criteria < lowest_criteria:
                otsu_threshold = threshold
                lowest_criteria = otsu_criteria

    return binarize_by_thresholding(img, otsu_threshold)


def binarize_by_dithering(img: np.ndarray) -> np.ndarray:
    """Returns a binary image by applying the Floyd–Steinberg dithering algorithm to a grayscale image."""
    # Add one extra row to avoid dealing with "corner cases" in the loop.
    padded_img = np.zeros(shape=(img.shape[0] + 1, img.shape[1] + 1), dtype=img.dtype)
    padded_img[:-1, :-1] = img
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # YOUR CODE HERE:
            #   Assume that img ranges from 0 to 255.
            #   ...
            value = padded_img[i, j]
            if value > 127:
                out[i, j] = 255
            error = value - out[i, j]
            padded_img[i, j + 1] += error * 7 / 16
            padded_img[i + 1, j - 1] += error * 3 / 16
            padded_img[i + 1, j] += error * 5 / 16
            padded_img[i + 1, j + 1] += error * 1 / 16
    return out


if __name__ == "__main__":
    # Show effect
    original_img = cv2.imread(sample_filepath('airplane.tiff'), cv2.IMREAD_GRAYSCALE)  # Read the image.
    original_img = original_img.astype('float32')  # Convert to float32 to avoid overflow and rounding errors

    for image in [original_img, original_img[175:225, 70:120]]:
        results = {
            'img': image,
            'Threshold_64': binarize_by_thresholding(image, 64),
            'Threshold_128': binarize_by_thresholding(image, 128),
            'Threshold_192': binarize_by_thresholding(image, 192),
            'Otsu': binarize_by_otsu(image),
            'Dithering': binarize_by_dithering(image),
        }
        results = {k: v for k, v in results.items() if v is not None}  # Remove None values.

        # Visualize images
        fig, axs = plt.subplots(2, 3)
        # Remove default axis
        for ax in axs.flatten():
            ax.axis('off')
        # Show one image per subplot
        for ax, (title, subimage) in zip(axs.flatten(), results.items()):
            ax.set_title(title)
            ax.imshow(subimage, cmap='gray')
        # Display figure
        plt.show()

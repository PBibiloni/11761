import cv2
import numpy as np
from matplotlib import pyplot as plt, colors

from utils import sample_filepath


def discrete_cosinus_transform(img: np.ndarray) -> np.ndarray:
    """Returns the dct coefficients of the image."""
    # YOUR CODE HERE:
    #   See `cv2.dct(...)`, assume img ranges from 0 to 1.
    #   ...


def invert_discrete_consinus_transform(img: np.ndarray) -> np.ndarray:
    """Returns the image from its dct coefficients."""
    # YOUR CODE HERE:
    #   See `cv2.dct(..., flags=...)`
    #   ...


def remove_last_coefficients(dct_coefficients: np.ndarray, remove_since_x: int, remove_since_y: int) -> np.ndarray:
    """Returns the dct coefficients of the image."""
    # YOUR CODE HERE:
    #   ...


def center_coefficients(dct_coefficients: np.ndarray) -> np.ndarray:
    """Returns a tensor where the coefficients have been switched so the origin is in the middle."""
    # YOUR CODE HERE:
    #   ...


if __name__ == "__main__":
    original_img = cv2.imread(sample_filepath('mandril.tiff'), cv2.IMREAD_GRAYSCALE).astype('float32') / 255.0
    # Direct and invert DCT transform
    dct_coefficients = discrete_cosinus_transform(original_img)
    shifted_dct_coefficients = center_coefficients(dct_coefficients)
    recovered_img = invert_discrete_consinus_transform(dct_coefficients)
    # Remove some coefficients and invert DCT transform
    removed_dct_coefficients = remove_last_coefficients(dct_coefficients, 30, 30)
    shifted_and_removed_dct_coefficients = center_coefficients(removed_dct_coefficients)
    filtered_img = invert_discrete_consinus_transform(removed_dct_coefficients)

    # Visualize the effect of computing and reverting the DCT
    fig, axs = plt.subplots(1, 4)
    fig.suptitle('Forward and inverse DCT')
    axs[0].imshow(original_img, cmap='gray')
    axs[1].imshow(np.abs(dct_coefficients), cmap='gray', norm=colors.PowerNorm(gamma=0.05))
    axs[2].imshow(np.abs(shifted_dct_coefficients), cmap='gray', norm=colors.PowerNorm(gamma=0.05))
    axs[3].imshow(recovered_img, cmap='gray')
    plt.show()

    # Visualize the effect of removing the last coefficients
    fig, axs = plt.subplots(1, 4)
    fig.suptitle('Removing last DCT coefficients')
    axs[0].imshow(original_img, cmap='gray')
    axs[1].imshow(np.abs(shifted_dct_coefficients), cmap='gray', norm=colors.PowerNorm(gamma=0.05))
    axs[2].imshow(np.abs(shifted_and_removed_dct_coefficients), cmap='gray', norm=colors.PowerNorm(gamma=0.05))
    axs[3].imshow(filtered_img, cmap='gray')
    plt.show()

import cv2
import numpy as np
from matplotlib import pyplot as plt


def discrete_cosinus_transform(img: np.ndarray) -> np.ndarray:
    """Returns the dct coefficients of the image."""
    # Your code here: see `cv2.dct(...)`, assume img ranges from 0 to 1.
    # ...
    return cv2.dct(img)


def invert_discrete_consinus_transform(img: np.ndarray) -> np.ndarray:
    """Returns the image from its dct coefficients."""
    # Your code here: see `cv2.dct(..., flags=...)`
    # ...
    return cv2.dct(img, flags=cv2.DCT_INVERSE)


def remove_last_coefficients(dct_coefficients: np.ndarray, remove_since_x: int, remove_since_y: int) -> np.ndarray:
    """Returns the dct coefficients of the image."""
    # Your code here
    # ...
    removed_dct_coefficients = np.copy(dct_coefficients)
    removed_dct_coefficients[remove_since_x:, ] = 0
    removed_dct_coefficients[:, remove_since_y:] = 0
    return removed_dct_coefficients


def center_coefficients(dct_coefficients: np.ndarray) -> np.ndarray:
    """Returns a tensor where the coefficients have been switched so the origin is in the middle."""
    # Your code here
    # ...
    sh = dct_coefficients.shape
    shifted_dct_coefficients = np.zeros_like(dct_coefficients)
    shifted_dct_coefficients[:sh[0]//2, :sh[1]//2] = dct_coefficients[sh[0]//2:, sh[1]//2:]
    shifted_dct_coefficients[sh[0]//2:, :sh[1]//2] = dct_coefficients[:sh[0]//2, sh[1]//2:]
    shifted_dct_coefficients[:sh[0]//2, sh[1]//2:] = dct_coefficients[sh[0]//2:, :sh[1]//2]
    shifted_dct_coefficients[sh[0]//2:, sh[1]//2:] = dct_coefficients[:sh[0]//2, :sh[1]//2]
    return shifted_dct_coefficients


if __name__ == "__main__":
    # Show effect
    original_img = cv2.imread('../samples/mandril.tiff', cv2.IMREAD_GRAYSCALE).astype('float32') / 255.0
    dct_coefficients = discrete_cosinus_transform(original_img)
    shifted_dct_coefficients = center_coefficients(dct_coefficients)
    removed_dct_coefficients = remove_last_coefficients(shifted_dct_coefficients, 30, 30)

    recovered_img = invert_discrete_consinus_transform(dct_coefficients)
    filtered_img = invert_discrete_consinus_transform(removed_dct_coefficients)

    results = {
        'Original': original_img,
        'DCT coefficients': dct_coefficients,
        'DCT coefficients (cent)': shifted_dct_coefficients,
        'Recovered': recovered_img,
    }
    fig, axs = plt.subplots(1, 4)
    # Show one image per subplot
    for ax, (name, subimage) in zip(axs.flatten(), results.items()):
        ax.imshow(subimage, cmap='gray')
    # Display figure
    plt.show()

    results = {
        'Original': original_img,
        'DCT coefficients (cent)': shifted_dct_coefficients,
        'DCT coefficients (cent, 30x30 removed)': removed_dct_coefficients,
        'Recovered': filtered_img,
    }
    fig, axs = plt.subplots(1, 4)
    # Show one image per subplot
    for ax, (name, subimage) in zip(axs.flatten(), results.items()):
        ax.imshow(subimage, cmap='gray')
    # Display figure
    plt.show()

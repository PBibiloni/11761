import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import sample_filepath


def additive_white_gaussian_noise(image: np.ndarray, std: float) -> np.ndarray:
    """Adds additive white Gaussian noise to an image."""
    # YOUR CODE HERE: 
    #   See `np.random.normal(...)`.
    #   ...
    return image + np.random.normal(0, std, image.shape)


def uniform_multiplicative_noise(image: np.ndarray, a: float, b: float) -> np.ndarray:
    """Adds uniform multiplicative noise to an image."""
    # YOUR CODE HERE:
    #   See `np.random.uniform(...)`.
    #   ...
    return image * np.random.uniform(a, b, image.shape)


def salt_and_pepper_noise(image: np.ndarray, p: float) -> np.ndarray:
    """Adds salt and pepper noise to an image."""
    # YOUR CODE HERE:
    #   See `np.random.uniform(...)` and `np.random.choice(...)`.
    #   ...
    image = np.copy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if np.random.uniform(0, 1) < p:
                if np.random.uniform(0, 1) <= 0.5:
                    image[i, j] = 0
                else:
                    image[i, j] = 255
    return image


def shot_noise(image: np.ndarray) -> np.ndarray:
    """Add shot noise to an image."""
    # YOUR CODE HERE:
    #   See `np.random.poisson(...)`.
    #   ...
    return np.random.poisson(image)


if __name__ == '__main__':
    # Show effect on boat image.
    img = cv2.imread(sample_filepath('boat.tiff'), cv2.IMREAD_GRAYSCALE)  # Read the image.
    img = img.astype('float32')  # Convert to float32 to avoid overflow and rounding errors
    results = {
        'img': img,
        'AWGN': additive_white_gaussian_noise(img, std=10.0),
        'uniform_multiplicative': uniform_multiplicative_noise(img, a=0.7, b=1.3),
        'salt_and_pepper': salt_and_pepper_noise(img, p=0.20),
        'shot_noise': shot_noise(img),
    }
    results = {k: v for k, v in results.items() if v is not None}  # Remove None values.

    # Visualize images
    fig, axs = plt.subplots(2, 3)
    # Remove default axis
    for ax in axs.flatten():
        ax.axis('off')
    # Show one image per subplot
    for ax, (title, img) in zip(axs.flatten(), results.items()):
        ax.set_title(title)
        ax.imshow(img, cmap='gray')
    # Display figure
    plt.show()
import numpy as np
import cv2
from matplotlib import pyplot as plt

from utils import sample_filepath


def convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Returns the convolution of one image with a specific kernel (with no padding)."""
    img_sz_x, img_sz_y = image.shape
    krn_sz_x, krn_sz_y = kernel.shape
    out_sz_x = img_sz_x - krn_sz_x + 1  # Why?
    out_sz_y = img_sz_y - krn_sz_y + 1  # Why?
    out = np.zeros(shape=(out_sz_x, out_sz_y), dtype=image.dtype)
    for i in range(out_sz_x):
        for j in range(out_sz_y):
            # YOUR CODE HERE:
            #   ...
            pass

    return out


if __name__ == '__main__':
    # Show effect on subset of boat image.
    img = cv2.imread(sample_filepath('boat.tiff'), cv2.IMREAD_GRAYSCALE)  # Read the image.
    img = img.astype('float32')     # Convert to float32 to avoid overflow and rounding errors
    img = img[150:200, 150:200]     # Select a small window

    kernels = {
        'identity': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        'shift_left': np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
        'smooth': np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
        'sharpen': np.array([[-1, -1, -1], [-1, 17, -1], [-1, -1, -1]]) / 9,
    }
    results = {
        name: convolution(img, kernel)
        for name, kernel in kernels.items()
    }
    results = {k: v for k, v in results.items() if v is not None}  # Remove None values.

    # Visualize images
    fig, axs = plt.subplots(2, 2)
    # Remove default axis
    for ax in axs.flatten():
        ax.axis('off')
    # Show one image per subplot
    for ax, (title, img) in zip(axs.flatten(), results.items()):
        ax.set_title(title)
        ax.imshow(img, cmap='gray')
    # Display figure
    plt.show()

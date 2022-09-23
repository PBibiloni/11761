import numpy as np
import cv2


def convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Sequence of limits of each bin (e.g. [0.0, 85.0, 170.0, 255.0] for 3 bins)."""
    # YOUR CODE HERE
    # ...
    img_sz_x, img_sz_y = image.shape
    krn_sz_x, krn_sz_y = kernel.shape
    out_sz_x = img_sz_x - krn_sz_x + 1  # Why?
    out_sz_y = img_sz_y - krn_sz_y + 1  # Why?
    out = np.zeros(shape=(out_sz_x, out_sz_y), dtype=image.dtype)
    for i in range(out_sz_x):
        for j in range(out_sz_y):
            # YOUR CODE HERE
            # ...
            for k in range(krn_sz_x):
                for l in range(krn_sz_y):
                    out[i, j] += image[i + (krn_sz_x-1) - k, j + (krn_sz_y-1) - l] * kernel[k, l]
    return out

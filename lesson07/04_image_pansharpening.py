import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import sample_filepath


def pansharpen_mean(panchromatic_img: np.ndarray, r_image: np.ndarray, g_image: np.ndarray, b_image: np.ndarray) -> np.ndarray:
    """ Return RGB pansharpened image using the "simple mean" method. """
    # YOUR CODE HERE:
    #   See cv2.resize(...)
    #   ...
    # Upsize images
    r_in = cv2.resize(r_image, (panchromatic_img.shape[1], panchromatic_img.shape[0]))
    g_in = cv2.resize(g_image, (panchromatic_img.shape[1], panchromatic_img.shape[0]))
    b_in = cv2.resize(b_image, (panchromatic_img.shape[1], panchromatic_img.shape[0]))
    # Apply mean with panchromatic information
    r_out = panchromatic_img//2 + r_in//2
    g_out = panchromatic_img//2 + g_in//2
    b_out = panchromatic_img//2 + b_in//2
    # Return RGB image
    return np.dstack((r_out, g_out, b_out))


def panshapen_Brovey(panchromatic_img: np.ndarray, r_image: np.ndarray, g_image: np.ndarray, b_image: np.ndarray) -> np.ndarray:
    """ Return RGB pansharpened image using the Brovey method. """
    # YOUR CODE HERE:
    #   ...
    # Upsize images
    r_in = cv2.resize(r_image, (panchromatic_img.shape[1], panchromatic_img.shape[0])).astype('float')
    g_in = cv2.resize(g_image, (panchromatic_img.shape[1], panchromatic_img.shape[0])).astype('float')
    b_in = cv2.resize(b_image, (panchromatic_img.shape[1], panchromatic_img.shape[0])).astype('float')
    # Compute normalization factor
    normalization = panchromatic_img / (r_in + g_in + b_in)
    # Apply mean with respect to color-normalization factor
    r_out = r_in * normalization
    g_out = g_in * normalization
    b_out = b_in * normalization
    # Return RGB image
    return np.dstack((r_out, g_out, b_out)).astype('uint8')


def pansharpen_replace_intensity(panchromatic_img: np.ndarray, r_image: np.ndarray, g_image: np.ndarray, b_image: np.ndarray) -> np.ndarray:
    """ Return RGB pansharpened image replacing the intensity, and preserving chromatic information. """
    # YOUR CODE HERE:
    #   What is the best "intensity" channel?
    #   ...
    # Upsize images
    r_in = cv2.resize(r_image, (panchromatic_img.shape[1], panchromatic_img.shape[0]))
    g_in = cv2.resize(g_image, (panchromatic_img.shape[1], panchromatic_img.shape[0]))
    b_in = cv2.resize(b_image, (panchromatic_img.shape[1], panchromatic_img.shape[0]))
    # Create RGB composition
    img_rgb = np.dstack((r_in, g_in, b_in))
    # Replace lightness with panchromatic information
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    img_lab[:, :, 0] = panchromatic_img
    # Return RGB image
    return cv2.cvtColor(img_lab, cv2.COLOR_Lab2RGB)


if __name__ == "__main__":
    img_bgr = cv2.imread(sample_filepath('mandril.tiff'), cv2.IMREAD_COLOR)
    # Create panchromatic and smaller B, G, R channels.
    img_panchromatic = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_bgr_small = cv2.resize(img_bgr, (0, 0), fx=0.5, fy=0.5)
    img_b = img_bgr_small[:, :, 0]
    img_g = img_bgr_small[:, :, 1]
    img_r = img_bgr_small[:, :, 2]

    results = {
        'Original': cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        'Mean': pansharpen_mean(img_panchromatic, img_r, img_g, img_b),
        'Brovey': panshapen_Brovey(img_panchromatic, img_r, img_g, img_b),
        'Replace intensity': pansharpen_replace_intensity(img_panchromatic, img_r, img_g, img_b),
    }
    results = {k: v for k, v in results.items() if v is not None}

    # Visualize images
    fig, axs = plt.subplots(2, 2)
    [ax.axis('off') for ax in axs.flatten()]
    # Show one image per subplot
    for ax, (title, subimage) in zip(axs.flatten(), results.items()):
        ax.imshow(subimage)
        ax.set_title(title)
    plt.show()
import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import sample_filepath


def pansharpen_mean(panchromatic_img: np.ndarray, r_image: np.ndarray, g_image: np.ndarray, b_image: np.ndarray) -> np.ndarray:
    """ Return RGB pansharpened image using the "simple mean" method. """
    # YOUR CODE HERE: see cv2.resize(...)
    # ...


def panshapen_Brovey(panchromatic_img: np.ndarray, r_image: np.ndarray, g_image: np.ndarray, b_image: np.ndarray) -> np.ndarray:
    """ Return RGB pansharpened image using the Brovey method. """
    # YOUR CODE HERE
    # ...


def pansharpen_replace_intensity(panchromatic_img: np.ndarray, r_image: np.ndarray, g_image: np.ndarray, b_image: np.ndarray) -> np.ndarray:
    """ Return RGB pansharpened image replacing the intensity, and preserving chromatic information. """
    # YOUR CODE HERE: what is the best "intensity" channel?
    # ...


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
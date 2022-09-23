import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import sample_filepath


def intensity(img_bgr: np.ndarray) -> np.ndarray:
    """ Compute the intensity of an image. """
    # YOUR CODE HERE
    # ...


def luma(img_bgr: np.ndarray) -> np.ndarray:
    """ Compute the intensity of an image. """
    # YOUR CODE HERE
    # ...


def value(img_bgr: np.ndarray) -> np.ndarray:
    """ Compute the intensity of an image. """
    # YOUR CODE HERE
    # ...


def lightness_from_hsl(img_bgr: np.ndarray) -> np.ndarray:
    """ Compute the intensity of an image. """
    # YOUR CODE HERE
    # ...


def lightness_from_cielab(img_bgr: np.ndarray) -> np.ndarray:
    """ Compute the intensity of an image. """
    # YOUR CODE HERE
    # ...


if __name__ == "__main__":
    img_bgr = cv2.imread(sample_filepath('peppers.tiff'), cv2.IMREAD_COLOR)

    results = {
        # 'original': img_bgr,
        'intensity': intensity(img_bgr),
        'luma': luma(img_bgr),
        'value': value(img_bgr),
        'L (HSL)': lightness_from_hsl(img_bgr),
        'L* (CIEL*a*b*)': lightness_from_cielab(img_bgr),
    }
    results = {k: v for k, v in results.items() if v is not None}

    # Visualize images
    fig, axs = plt.subplots(1, len(results))
    [ax.axis('off') for ax in axs.flatten()]
    # Show one image per subplot
    for ax, (title, subimage) in zip(axs.flatten(), results.items()):
        ax.imshow(cv2.cvtColor(subimage, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
    plt.show()

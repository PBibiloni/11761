import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import sample_filepath


def process_on_best_channel(img_bgr: np.ndarray) -> np.ndarray:
    """ Apply a Canny edge detector on the `best` channel. """
    best_channel = img_bgr[:, :, 1]  # Why 1?
    # YOUR CODE HERE:
    #   Remember `cv2.Canny(...)`.
    #   ...


def process_on_intensity_channel(img_bgr: np.ndarray) -> np.ndarray:
    """ Apply a Canny edge detector on the intensity channel. """
    # YOUR CODE HERE:
    #   What is the best `intensity` channel?
    #   ...


def parallel_channels_then_combine(img_bgr: np.ndarray) -> np.ndarray:
    """ Apply a Canny edge detector on each channel, then combine them. """
    # YOUR CODE HERE:
    #   See cv2.bitwise_or(...), cv2.bitwise_and(...)
    #   ...


if __name__ == "__main__":
    img_bgr = cv2.imread(sample_filepath('peppers.tiff'), cv2.IMREAD_COLOR)

    results = {
        'original': img_bgr,
        'Best channel': process_on_best_channel(img_bgr),
        'Intensity': process_on_intensity_channel(img_bgr),
        'In parallel then combine': parallel_channels_then_combine(img_bgr),
    }
    results = {k: v for k, v in results.items() if v is not None}

    # Visualize images
    fig, axs = plt.subplots(2, 2)
    [ax.axis('off') for ax in axs.flatten()]
    # Show one image per subplot
    for ax, (title, subimage) in zip(axs.flatten(), results.items()):
        ax.imshow(cv2.cvtColor(subimage, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
    plt.show()

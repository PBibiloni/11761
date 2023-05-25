import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import sample_filepath


def process_in_parallel(img: np.ndarray) -> np.ndarray:
    """ Apply a histogram equalization to all channels independently. """
    # YOUR CODE HERE:
    #   See cv2.dilate(...)
    #   ...

def process_intensity_channel_preserve_chroma(img: np.ndarray) -> np.ndarray:
    """ Apply a histogram equalization to intensity channel only. """
    # YOUR CODE HERE:
    #   What is the best intensity-and-chroma color space?
    #   ...


if __name__ == "__main__":
    img_bgr = cv2.imread(sample_filepath('peppers.tiff'), cv2.IMREAD_COLOR)

    results = {
        # 'original': img_bgr,
        'In parallel then combine': process_in_parallel(img_bgr),
        'Intensity': process_intensity_channel_preserve_chroma(img_bgr),
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
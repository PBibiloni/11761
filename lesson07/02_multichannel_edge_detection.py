import cv2
import numpy as np
from matplotlib import pyplot as plt


def process_on_best_channel(img_bgr: np.ndarray) -> np.ndarray:
    """ Apply a Canny edge detector on the `best` channel. """
    best_channel = img_bgr[:, :, 1]  # Why 1?
    # YOUR CODE HERE: see cv2.Canny(...)
    # ...
    return cv2.Canny(best_channel, 100, 200)


def process_on_intensity_channel(img_bgr: np.ndarray) -> np.ndarray:
    """ Apply a Canny edge detector on the intensity channel. """
    # YOUR CODE HERE: what is the best `intensity` channel?
    # ...
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return cv2.Canny(img_lab[:, :, 0], 100, 200)


def parallel_channels_then_combine(img_bgr: np.ndarray) -> np.ndarray:
    """ Apply a Canny edge detector on each channel, then combine them. """
    # YOUR CODE HERE: see cv2.bitwise_or(...), cv2.bitwise_and(...)
    # ...
    all_edges = np.zeros(shape=img_bgr.shape[:2], dtype=img_bgr.dtype)
    for band in range(img_bgr.shape[-1]):
        all_edges = np.bitwise_or(all_edges, cv2.Canny(img_bgr[:, :, band], 100, 200))
    return all_edges


if __name__ == "__main__":
    img_bgr = cv2.imread(sample_filepath('samples/peppers.tiff'), cv2.IMREAD_COLOR)

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

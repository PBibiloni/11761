import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import sample_filepath


def histogram_find_cuts(nbins: int) -> np.ndarray:
    """Sequence of evenly-spaced limits of each bin (e.g. [0.0, 85.0, 170.0, 255.0] for 3 bins)."""
    # YOUR CODE HERE:
    #   See `np.linspace(...)` and `np.arange(...)`.
    #   ...
    return np.arange(nbins + 1) * 255 / (nbins)


def histogram_count_values(image: np.ndarray, nbins: int) -> np.ndarray:
    """Creates a histogram of a grayscale image."""
    size_x = image.shape[0]
    size_y = image.shape[1]
    hist = np.zeros(nbins)  # Variable to store the histogram, initialized at 0.
    for i in range(size_x):
        for j in range(size_y):
            value = image[i, j]
            # YOUR CODE HERE:
            #   ...
            discretized_value = int(value * nbins / 255)
            hist[discretized_value] += 1
    return hist

def histogram_plot(image: np.ndarray, nbins) -> None:
    """Plots a histogram of a grayscale image."""
    # YOUR CODE HERE:
    #   Use a bar plot `plt.bar(...)`.
    #   ...
    cuts = histogram_find_cuts(nbins=nbins)
    values = histogram_count_values(image, nbins=nbins)

    centers = (cuts[:-1] + cuts[1:]) / 2
    plt.bar(centers, values, align='center', width=cuts[1]-cuts[0])
    if len(cuts) <= 30:
        plt.xticks(cuts)
    plt.show()


if __name__ == '__main__':
    # Load the image
    image = cv2.imread(sample_filepath('tank.tiff'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(image, cmap='gray')
    plt.show()

    # Create the histograms
    for n in [8, 16, 32]:
        cuts = histogram_find_cuts(nbins=n)
        print(f'# For {n} bins:')
        print(f'Histogram bins are separated by: {"-".join(f"{c:.1f}" for c in np.nditer(cuts))}.')

        values = histogram_count_values(image, nbins=n)
        for start, end, value in zip(cuts[:-1], cuts[1:], values):
            print(f'[{start:5.1f}, {end:5.1f}): {value}')

        histogram_plot(image, nbins=n)

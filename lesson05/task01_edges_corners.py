import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import sample_filepath


def find_edges(img_gray: np.ndarray) -> np.ndarray:
    """Adjust the Canny edge detector to find the edges in one image."""
    # YOUR CODE HERE:
    #   See `cv2.Canny(img_gray, ...)` with the `img_gray` as input.
    #   ...


def find_corners(img_gray: np.ndarray) -> np.ndarray:
    """Adjust the Harris corner detector to find the corners in one image."""
    # YOUR CODE HERE: 
    #   See `cv2.cornerHarris(img_gray, ...)` with the `img_gray` as input.
    #   Use `cv2.dilate(...)` to remove non-local maxima.
    #   ...


if __name__ == "__main__":
    img = cv2.imread(sample_filepath('cathedral.jpg'), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edges = find_edges(img_gray)
    img_corners = find_corners(img_gray)

    img_with_edges = img.copy()
    img_with_edges[img_edges == 255, :] = [0, 255, 0]
    img_with_corners = img.copy()
    img_with_corners[cv2.dilate(img_corners, np.ones((3, 3))) == 255] = [255, 0, 0]
    # Visualize images
    fig, axs = plt.subplots(2, 2)
    # Show one image per subplot
    axs[0, 0].set_title('Original')
    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title('Grayscale')
    axs[0, 1].imshow(img_gray, cmap='gray')
    axs[1, 0].set_title('Edges')
    axs[1, 0].imshow(cv2.cvtColor(img_with_edges, cv2.COLOR_BGR2RGB), cmap='gray')
    axs[1, 1].set_title('Corners')
    axs[1, 1].imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB), cmap='gray')
    # Display figure
    plt.show()

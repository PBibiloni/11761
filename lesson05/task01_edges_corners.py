import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import sample_filepath


def find_edges(img_gray: np.ndarray) -> np.ndarray:
    """Adjust the Canny edge detector to find the edges in one image."""
    # Your code here: see `cv2.Canny(img_gray, ...)` with the `img_gray` as input.
    # ...
    img_edges = cv2.Canny(img_gray, threshold1=100, threshold2=200, apertureSize=3)
    return img_edges


def find_corners(img_gray: np.ndarray) -> np.ndarray:
    """Adjust the Harris corner detector to find the corners in one image."""
    # Your code here: see `cv2.cornerHarris(img_gray, ...)` with the `img_gray` as input.
    #                 use `cv2.dilate(...)` to remove non-local maxima.
    # ...
    # Create a `corner` value for each pixel
    corner_map = cv2.cornerHarris(img_gray, blockSize=5, ksize=3, k=0.1)
    # Set img_corners[x, y] = 255 if (x, y) is local maxima, set to 0 otherwise
    corner_map_maxneighbourhood = cv2.dilate(corner_map, np.ones((11, 11)))
    img_corners = np.zeros_like(img_gray)
    # Return value for visualization, whose pixels should be either 0 o 255.
    img_corners[corner_map == corner_map_maxneighbourhood] = 255
    return img_corners


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

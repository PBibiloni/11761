import cv2
import numpy as np
from matplotlib import pyplot as plt

from lesson03.task01_binarize import binarize_by_thresholding
from utils import sample_filepath


def label_connected_components(binary_img: np.ndarray) -> np.ndarray:
    """Returns a labeled version of the image, where each connected component is assigned a different label."""
    label_img = np.zeros_like(binary_img, dtype=np.uint16)
    collisions = {}     # { label: min_label_in_same_CC }
    for i in range(binary_img.shape[0]):
        for j in range(binary_img.shape[1]):
            if binary_img[i, j]:
                previous_labels = []
                # We use a 4-neighbour connectivity to find out previous labelled pixels
                if i >= 1 and label_img[i-1, j]:
                    previous_labels.append(label_img[i-1, j])
                if j >= 1 and label_img[i, j-1]:
                    previous_labels.append(label_img[i, j-1])

                # YOUR CODE HERE:
                #   Label each pixel and create a collision dictionary.
                #   ...
                if len(previous_labels) == 0:
                    # No labelled neighbours: create a new label
                    label_img[i, j] = np.max(label_img) + 1
                elif len(previous_labels) == 1:
                    # One labelled neighbour: use their label
                    label_img[i, j] = min(previous_labels)
                else:
                    # Multiple labelled neighbours
                    # Find minimum label in current connected component.
                    representative_label = min(previous_labels)
                    for label in previous_labels:
                        if label in collisions:
                            representative_label = min(representative_label, collisions[label])
                    # Assign current pixel and update collisions dictionary.
                    label_img[i, j] = representative_label
                    for label in previous_labels:
                        collisions[label] = representative_label

    # Make collision dictionary transitive.
    for label in range(np.max(label_img)):  # Ordered lookup is important here.
        if label in collisions:
            representative = collisions[label]
            # If representative is not a root, find its root.
            if representative in collisions:
                collisions[label] = collisions[representative]

    # Replace labels with their representatives.
    for label, min_label_in_same_cc in collisions.items():
        # YOUR CODE HERE:
        #   ...
        label_img[label_img == label] = min_label_in_same_cc
    return label_img


if __name__ == "__main__":
    # Show effect
    img = cv2.imread(sample_filepath('airplane.tiff'), cv2.IMREAD_GRAYSCALE)  # Read the image.
    img = img.astype('float32')  # Convert to float32 to avoid overflow and rounding errors

    # Visualize CCs for different thresholds
    fig, axs = plt.subplots(2, 2)
    # Show one image per subplot
    axs.flatten()[0].set_title('Orginal')
    axs.flatten()[0].imshow(img, cmap='gray')
    for ax, th in zip(axs.flatten()[1:], [176, 192, 208]):
        ax.set_title(f'Threshold {th:d} -> CCs')
        ax.imshow(
            label_connected_components(binarize_by_thresholding(img, th)),
            cmap='Pastel1')
    # Display figure
    plt.show()

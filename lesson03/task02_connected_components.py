import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

from lesson03.task01_binarize import binarize_by_thresholding


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
                # ...
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
    for label, min_label_in_same_cc in collisions.items():
        # YOUR CODE HERE: see `np.where(...)`.
        # ...
        label_img[label_img == label] = min_label_in_same_cc
    return label_img


def binarize_by_hysteresis(img: np.ndarray, low_threshold: float, high_threshold: float) -> np.ndarray:
    """Returns a binary version of the image by applying a hysteresis operation."""
    out = np.zeros_like(img)
    label_img = label_connected_components(binarize_by_thresholding(img, low_threshold))
    labels = np.unique(label_img)
    for label in labels:
        # YOUR CODE HERE: see `np.any(...)` and `np.all(...)`.
        # ...
        if np.any(img[label_img == label] >= high_threshold):
            out[label_img == label] = 255
    return out


if __name__ == "__main__":
    # Show effect
    img = cv2.imread('../samples/airplane.tiff', cv2.IMREAD_GRAYSCALE)  # Read the image.
    img = img.astype('float32')  # Convert to float32 to avoid overflow and rounding errors
    # img = cv2.resize(img, dsize=(img.shape[1]//20, img.shape[0]//20))

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

    # Visualize Hystersis for different upper thresholds
    fig, axs = plt.subplots(2, 2)
    # Remove default axis
    for ax in axs.flatten():
        ax.axis('off')
    # Show one image per subplot
    axs.flatten()[0].set_title('Orginal')
    axs.flatten()[0].imshow(img, cmap='gray')
    for ax, upper_th in zip(axs.flatten()[1:], [192, 208, 224]):
        ax.set_title(f'Hysteresis (192, {upper_th:d})')
        ax.imshow(
            binarize_by_hysteresis(img, low_threshold=128, high_threshold=upper_th),
            cmap='gray')
    # Display figure
    plt.show()


    # Save images
    os.makedirs('../results', exist_ok=True)
    out = binarize_by_hysteresis(img, low_threshold=128, high_threshold=192)
    cv2.imwrite(f'../results/task02_hyst_128_192.png', out.astype('uint8'))
    cv2.imwrite(f'../results/task02_hyst_128_192_detail.png', out[175:225, 70:120].astype('uint8'))


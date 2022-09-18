import typing as tp

import cv2
import numpy as np
from matplotlib import pyplot as plt


def segmentation_by_watershed(img_bgr: np.ndarray, seed_pixel: tp.Tuple[int, int]) -> np.ndarray:
    """Segment the image by considering a watershed method."""
    # Smooth image to improve results.
    img_smoothed = cv2.GaussianBlur(img_bgr, ksize=(5, 5), sigmaX=0)
    # Initialize markers (mark as 0 unknown pixels, as 1 background, as 2 foreground)
    markers = np.ones(img_bgr.shape[:2], dtype=np.int32)
    offset = 30
    sh = img_bgr.shape
    markers[max(seed_pixel[0]-offset, 0):min(seed_pixel[0]+offset, sh[0]), max(seed_pixel[1]-offset, 0):min(seed_pixel[1]+offset, sh[1])] = 0
    markers[seed_pixel] = 2
    # Apply watershed transformation
    watshd = cv2.watershed(img_smoothed, markers=markers)
    _, axs = plt.subplots(2,2)
    axs[0, 0].imshow(img_bgr)
    axs[0, 1].imshow(markers*128, cmap='gray')
    axs[1, 0].imshow(watshd, cmap='gray')
    plt.show()
    return watshd


def contour_based_segmentation(img_gray: np.ndarray, seed_pixel: tp.Tuple[int, int]) -> np.ndarray:
    """Segment the image by considering contours derived from edges."""
    img_edges = cv2.Canny(img_gray, threshold1=100, threshold2=200, apertureSize=3)
    contours, hierarchy = cv2.findContours(img_edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    # Select only the contour containing the seed point
    for contour in contours:
        region_mask = np.zeros_like(img_gray)
        # Draw the contour and its interior on the mask
        region_mask = cv2.drawContours(region_mask, [contour], contourIdx=-1, color=255, thickness=-1)  # Draw Interior
        region_mask = cv2.dilate(region_mask, kernel=np.ones((3, 3)))   # Add border too
        if region_mask[seed_pixel]:
            return region_mask

    return np.zeros_like(img_gray)


if __name__ == "__main__":
    img_bgr = cv2.imread('../samples/Ki67.jpg', cv2.IMREAD_COLOR)
    img_bgr = cv2.resize(img_bgr, (0, 0), fx=0.25, fy=0.25)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Detect region of interest (where brown cells might should be present)
    region_of_interest = cv2.erode((img_gray <= 128).astype('uint8')*255, np.ones((3, 3)))
    # Randomly select a seed within the region of interest
    positive_points = np.where(region_of_interest != 0)
    seed_idx = np.random.choice(len(positive_points[0]))
    seed_point = positive_points[0][seed_idx], positive_points[1][seed_idx]

    results = {
        'Region of interest': region_of_interest,
        'Segmentation by watershed': segmentation_by_watershed(img_bgr, seed_point),
        'Edge-based segmentation': contour_based_segmentation(img_gray, seed_point),
    }
    results = {k: v for k, v in results.items() if v is not None}

    # Visualize images
    fig, axs = plt.subplots(2, 2)
    [ax.axis('off') for ax in axs.flatten()]
    # Show one image per subplot
    for ax, (title, binary_image) in zip(axs.flatten(), results.items()):
        subimage = np.copy(img_bgr)
        subimage[binary_image != cv2.erode(binary_image.astype('uint8'), np.ones((3, 3))), ...] = (0, 255, 0)
        cv2.line(subimage, pt1=(seed_point[1] - 5, seed_point[0] - 5), pt2=(seed_point[1] + 5, seed_point[0] + 5),
                 color=(0, 0, 255), thickness=1)
        cv2.line(subimage, pt1=(seed_point[1] - 5, seed_point[0] + 5), pt2=(seed_point[1] + 5, seed_point[0] - 5),
                 color=(0, 0, 255), thickness=1)
        ax.imshow(cv2.cvtColor(subimage, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
    plt.show()

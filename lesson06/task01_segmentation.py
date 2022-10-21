import typing as tp

import cv2
import numpy as np
from matplotlib import pyplot as plt


def segmentation_by_watershed(img_bgr: np.ndarray, seed_pixel: tp.Tuple[int, int]) -> np.ndarray:
    """Segment the image by considering a watershed method."""
    # Your code here: see cv2.watershed(...)
    # ...
    markers = np.ones(img_bgr.shape[:2], dtype=np.int32)
    markers[10:-10, 10:-10] = 0
    markers[seed_pixel] = 255
    watshd = cv2.watershed(img_bgr, markers=markers)
    return watshd


def edge_based_segmentation(img_gray: np.ndarray, seed_pixel: tp.Tuple[int, int]) -> np.ndarray:
    """Segment the image by considering contours derived from edges."""
    # Your code here: see cv2.Canny(...) and cv2.findContours(...).
    # ...
    segmentation = np.zeros_like(img_gray)
    img_edges = cv2.Canny(img_gray, threshold1=100, threshold2=200, apertureSize=3)
    contours, hierarchy = cv2.findContours(img_edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    # Your code here: Draw the contours on a black image
    for contour in contours:
        pointPolygonTest_seed = np.array(seed_pixel, dtype='uint8')
        if cv2.pointPolygonTest(contour, pointPolygonTest_seed, measureDist=False) != -1:
            cv2.drawContours(segmentation, [contour], contourIdx=-1, color=255, thickness=-1)

    return segmentation


def region_growing_segmentation(img_gray: np.ndarray, seed_pixel: tp.Tuple[int, int]) -> np.ndarray:
    """Segment the image by considering a region-growing method."""
    # Your code here:
    # ...
    segmentation = np.zeros_like(img_gray)
    current_pixels = [seed_pixel]
    while current_pixels:
        pixel = current_pixels.pop()
        # Your code here: Add the pixel to the segmentation
        # ...
        segmentation[pixel] = 255

        # Your code here: Add [some of] its neighbours to the list of current pixels
        # ...
        candidates = [(pixel[0] + 1, pixel[1]), (pixel[0] - 1, pixel[1]), (pixel[0], pixel[1] + 1), (pixel[0], pixel[1] - 1)]
        for candidate in candidates:
            if candidate[0] >= 0 and candidate[0] < img_gray.shape[0] and candidate[1] >= 0 and candidate[1] < img_gray.shape[1]:
                if segmentation[candidate] == 0 and abs(img_gray[pixel] - img_gray[candidate]) < 50:
                    current_pixels.append(candidate)

        # Check for an ending condition
        if np.sum(segmentation) > 0.3 * img_gray.size:
            break

    return segmentation


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
        'Edge-based segmentation': edge_based_segmentation(img_gray, seed_point),
        'Region-growing segmentation': region_growing_segmentation(img_gray, seed_point),
    }
    results = {k: v for k, v in results.items() if v is not None}

    # Visualize images
    fig, axs = plt.subplots(2, 2)
    # Show one image per subplot
    for ax, (title, binary_image) in zip(axs.flatten(), results.items()):
        subimage = np.copy(img_bgr)
        subimage[binary_image != cv2.erode(binary_image.astype('uint8'), np.ones((3, 3))), ...] = (0, 255, 0)
        cv2.line(subimage, pt1=(seed_point[1] - 5, seed_point[0] - 5), pt2=(seed_point[1] + 5, seed_point[0] + 5),
                 color=(0, 0, 255), thickness=2)
        cv2.line(subimage, pt1=(seed_point[1] - 5, seed_point[0] + 5), pt2=(seed_point[1] + 5, seed_point[0] - 5),
                 color=(0, 0, 255), thickness=2)
        ax.imshow(cv2.cvtColor(subimage, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
    # axs[0, 0].set_title('Original')
    # axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # axs[0, 1].set_title('Grayscale')
    # axs[0, 1].imshow(img_gray, cmap='gray')
    # axs[1, 0].set_title('Edges')
    # axs[1, 0].imshow(cv2.cvtColor(img_with_edges, cv2.COLOR_BGR2RGB), cmap='gray')
    # axs[1, 1].set_title('Corners')
    # axs[1, 1].imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB), cmap='gray')
    # Display figure
    plt.show()

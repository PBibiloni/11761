import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import sample_filepath


def intensity(img_bgr: np.ndarray) -> np.ndarray:
    """ Compute the intensity of an image. """
    # YOUR CODE HERE:
    #   ...
    img_bgr = img_bgr.astype('float')
    img_gray = (img_bgr[:, :, 0] + img_bgr[:, :, 1] + img_bgr[:, :, 2]) / 3
    return img_gray.astype('uint8')


def luma(img_bgr: np.ndarray) -> np.ndarray:
    """ Compute the intensity of an image. """
    # YOUR CODE HERE:
    #   ...
    img_bgr = img_bgr.astype('float')
    img_gray = 0.2126*img_bgr[:, :, 2] + 0.7152*img_bgr[:, :, 1] + 0.0722*img_bgr[:, :, 0]
    return img_gray.astype('uint8')


def value(img_bgr: np.ndarray) -> np.ndarray:
    """ Compute the intensity of an image. """
    # YOUR CODE HERE:
    #   ...
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return img_hsv[:, :, 2]


def lightness_from_hsl(img_bgr: np.ndarray) -> np.ndarray:
    """ Compute the intensity of an image. """
    # YOUR CODE HERE:
    #   ...
    img_hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    return img_hls[:, :, 1]


def lightness_from_cielab(img_bgr: np.ndarray) -> np.ndarray:
    """ Compute the intensity of an image. """
    # YOUR CODE HERE:
    #   ...
    img_hsl = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    return img_hsl[:, :, 0]


if __name__ == "__main__":
    img_bgr = cv2.imread(sample_filepath('peppers.tiff'), cv2.IMREAD_COLOR)

    results = {
        # 'original': img_bgr,
        'intensity': intensity(img_bgr),
        'luma': luma(img_bgr),
        'value': value(img_bgr),
        'L (HSL)': lightness_from_hsl(img_bgr),
        'L* (CIEL*a*b*)': lightness_from_cielab(img_bgr),
    }
    results = {k: v for k, v in results.items() if v is not None}

    # Visualize images
    fig, axs = plt.subplots(2, 3)
    [ax.axis('off') for ax in axs.flatten()]
    # Show one image per subplot
    for ax, (title, subimage) in zip(axs.flatten(), results.items()):
        ax.imshow(cv2.cvtColor(subimage, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
    plt.show()

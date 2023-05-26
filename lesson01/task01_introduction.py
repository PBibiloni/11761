import cv2
from matplotlib import pyplot as plt

from utils import sample_filepath

img_filepath = sample_filepath('mandril.tiff')
print(img_filepath)
# YOUR CODE HERE:
#   Load the `mandril.tiff` sample image with OpenCV.
#   See `cv2.imread(...)`.
#   ...


# YOUR CODE HERE:
#   Print dtype, size (in pixels), and number of channels of the image.
#   See `[ndarray].dtype` and `[ndarray].shape`.
#   ...


# YOUR CODE HERE:
#   Transform from BGR to RGB and Grayscale (see `cv2.cvtColor(...)`, `cv2.COLOR_BGR2RGB`, `cv2.COLOR_BGR2GRAY`).
#   See `cv2.cvtColor(...)` and the flags `cv2.COLOR_BGR2RGB` and `cv2.COLOR_BGR2GRAY`.
#   ...


# YOUR CODE HERE:
#   Visualize with OpenCV.
#   See `cv2.imshow(...)`, `cv2.waitKey(...)` and `cv2.destroyAllWindows(...)`.
#   ...


# YOUR CODE HERE:
#   Visualize with Matplotlib.
#   See `plt.imshow(...)` and `plt.show()`.
#   ...


# YOUR CODE HERE:
#   Load video from file, and display it using OpenCV.
#   See `cv2.VideoCapture(...)`.
#   ...


# YOUR CODE HERE:
#   If available, load video from webcam, and display it using OpenCV
#   Use `cv2.VideoCapture(0)` to select the webcam.
#   ...

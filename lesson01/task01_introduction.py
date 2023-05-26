import cv2
from matplotlib import pyplot as plt

from utils import sample_filepath

img_filepath = sample_filepath('mandril.tiff')
print(img_filepath)
# YOUR CODE HERE:
#   Load the `mandril.tiff` sample image with OpenCV.
#   See `cv2.imread(...)`.
#   ...
img_bgr = cv2.imread(img_filepath)

# YOUR CODE HERE:
#   Print dtype, size (in pixels), and number of channels of the image.
#   See `[ndarray].dtype` and `[ndarray].shape`.
#   ...
print(f'img.dtype: {img_bgr.dtype}')
print(f'size in pixels: {img_bgr.shape[0]} x {img_bgr.shape[1]}')
print(f'number of channels: {img_bgr.shape[2]}')

# YOUR CODE HERE:
#   Transform from BGR to RGB and Grayscale (see `cv2.cvtColor(...)`, `cv2.COLOR_BGR2RGB`, `cv2.COLOR_BGR2GRAY`).
#   See `cv2.cvtColor(...)` and the flags `cv2.COLOR_BGR2RGB` and `cv2.COLOR_BGR2GRAY`.
#   ...
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # Also possible: img_rgb = img[..., ::-1]
img_grayscale = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# YOUR CODE HERE:
#   Visualize with OpenCV.
#   See `cv2.imshow(...)`, `cv2.waitKey(...)` and `cv2.destroyAllWindows(...)`.
#   ...
cv2.imshow('Title here', img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

# YOUR CODE HERE:
#   Visualize with Matplotlib.
#   See `plt.imshow(...)` and `plt.show()`.
#   ...
plt.imshow(img_rgb)
plt.show()

# YOUR CODE HERE:
#   Load video from file, and display it using OpenCV.
#   See `cv2.VideoCapture(...)`.
#   ...
video_filepath = sample_filepath('portitxol.mp4')
cap = cv2.VideoCapture(video_filepath)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Frame title here', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

# YOUR CODE HERE:
#   If available, load video from webcam, and display it using OpenCV
#   Use `cv2.VideoCapture(0)` to select the webcam.
#   ...
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Frame title here', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

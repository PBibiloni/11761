import time

import cv2
from matplotlib import pyplot as plt

from utils import sample_filepath

# 1) Load the `mandril` image with OpenCV (see `cv2.imread(...)`)
img_filepath = sample_filepath('mandril.tiff')
img_bgr = cv2.imread(img_filepath )

# 2) Find dtype, size (in pixels), and number of channels of the image (see `[img].dtype` and `[img].shape`)
print(f'img.dtype: {img_bgr.dtype}')
print(f'size in pixels: {img_bgr.shape[0]} x {img_bgr.shape[1]}')
print(f'number of channels: {img_bgr.shape[2]}')

# 3) Transform from BGR to RGB and Grayscale (see `cv2.cvtColor(...)`, `cv2.COLOR_BGR2RGB`, `cv2.COLOR_BGR2GRAY`)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# Also possible: img_rgb = img[..., ::-1]
img_grayscale = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# 4) Visualize with OpenCV (see `cv2.imshow(...)`, `cv2.waitKey(...)` and `cv2.destroyAllWindows(...)`)
cv2.imshow('Title here', img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 5) Visualize with Matplotlib (see `plt.imshow(...)` and `plt.show()`)
plt.imshow(img_rgb)
plt.show()

# 6) Load video from file, and display it using OpenCV (see `cv2.VideoCapture(...)`)
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

# 7) If available, load video from webcam, and display it using OpenCV (use `cv2.VideoCapture(0)`)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Frame title here', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

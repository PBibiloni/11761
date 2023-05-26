import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from utils import sample_filepath


def train_and_test_model():
    """ Trains and tests a model to classify superpixels. """
    # Read sample image
    img_bgr = cv2.imread(sample_filepath('Ki67.jpg'), cv2.IMREAD_COLOR)
    img_bgr = cv2.resize(img_bgr, (0, 0), fx=0.25, fy=0.25)
    # Infer (automatically) which pixels will be set as positives.
    ground_truth = cv2.dilate((cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) <= 128).astype('uint8') * 255, np.ones((3, 3)))

    # Superpixel segmentation
    sh = img_bgr.shape
    seeds = cv2.ximgproc.createSuperpixelSEEDS(image_width=sh[1], image_height=sh[0], image_channels=sh[2], num_superpixels=150, num_levels=4)
    seeds.iterate(img=img_bgr, num_iterations=5)
    region_labels = seeds.getLabels()

    # Get features
    # YOUR CODE HERE:
    #   Initialize the features (X) as one of the following (which is better?):
    #   >> X = get_geometric_features(region_labels), or
    #   >> X = get_photometric_features(img_bgr, region_labels), or
    #   >> X = np.concatenate([get_geometric_features(region_labels), get_photometric_features(img_bgr, region_labels)], axis=1)
    #   ...
    X = np.zeros((np.max(region_labels), 1))  # Features initialized as a single 0-vector (delete this line)

    # Get labels
    positives_per_region = []
    for idx_l in range(np.max(region_labels) + 1):
        region_mask = (region_labels == idx_l)
        positives_per_region.append(
            np.mean(ground_truth[region_mask])
        )
    y = np.array(positives_per_region) > 0.5

    # Create model
    # YOUR CODE HERE:
    #   Select a model from sklearn, e.g.:
    #   >> model = LogisticRegression(), or
    #   >> model = RandomForestClassifier().
    #   ...
    model = LogisticRegression()

    # Train model
    # YOUR CODE HERE:
    #   Use `model.fit(...)` to train the model.
    #   ...
    model.fit(X, y)

    # Make predictions
    # YOUR CODE HERE:
    #   Use `model.predict(X)` to train the model.
    #   ...
    predictions = np.zeros(img_bgr.shape[0:2])  # Initialize predictions as a 0-vector (delete this line).

    # Better visualizations
    border_mask = region_labels != cv2.erode(region_labels.astype('uint8'), np.ones((3, 3)))
    img_with_borders = np.copy(img_bgr)
    img_with_borders[border_mask, ...] = (255, 0, 0)
    img_with_gt = np.copy(img_bgr)
    img_with_gt[ground_truth == 255] = (0, 0, 255)
    img_with_gt[border_mask, ...] = (255, 0, 0)
    img_with_pred = np.copy(img_bgr)
    img_with_pred[predictions == 1] = (0, 255, 0)
    img_with_pred[border_mask, ...] = (255, 0, 0)

    _, axs = plt.subplots(2, 2)
    [ax.axis('off') for ax in axs.flatten()]
    axs[0, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axs[0, 1].imshow(cv2.cvtColor(img_with_borders, cv2.COLOR_BGR2RGB))
    axs[1, 0].imshow(cv2.cvtColor(img_with_gt, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title('Ground truth')
    axs[1, 1].imshow(cv2.cvtColor(img_with_pred, cv2.COLOR_BGR2RGB))
    axs[1, 1].set_title('Predictions')
    plt.show()


def get_geometric_features(region_labels:np.ndarray) -> np.ndarray:
    """ Computes geometric features for each region in the image. """
    features = []
    for label_idx in range(np.max(region_labels) + 1):
        region_mask = (region_labels == label_idx)

        # Compute geometric features for this region.
        area = np.sum(region_mask)
        # YOUR CODE HERE:
        #   Add other features (e.g. perimeter, centroid, roundness, ...)
        #   ...

        # Store them
        features.append([
            area,
            # YOUR CODE HERE:
            #   Append all features
            #   ...
        ])

    # Output as a numpy array indexed as [region_idx, feature_idx].
    X = np.array(features)
    return X


def get_photometric_features(img_bgr: np.ndarray, region_labels: np.ndarray) -> np.ndarray:
    """ Computes photometric features for each region in the image. """
    features = []
    for label_idx in range(np.max(region_labels) + 1):
        region_mask = (region_labels == label_idx)

        # Compute photometric features for this region.
        max_red_value = img_bgr[region_mask, 2].max()
        # YOUR CODE HERE:
        #   Add more features (e.g. other channels, mean values, std of values, ...)
        #   ...

        # Store them
        features.append([
            max_red_value,
            # YOUR CODE HERE:
            #   Append all features
            #   ...
        ])

    # Output as a numpy array indexed as [region_idx, feature_idx].
    X = np.array(features)
    return X


if __name__ == '__main__':
    train_and_test_model()


from pathlib import Path

import np as np
from PIL import Image
import numpy as np
from descriptors import *
from distances import *
from retrieval import retrieve
import pickle
from tqdm import tqdm
from bg_removal import *
import os
from text_detection import *
from noise_removal import *
from similar_artist import *
from text_combination import *

# set paths
QUERY_IMG_DIR = Path(os.path.join("..","data", "Week1", "qst1_w1"))
sift_extractor = SIFTExtractor()
hog_extractor = HOGExtractor()
color_sift_extractor = ColorSIFTExtractor()
gloh_extractor = GLOHExtractor()
daisy_extractor = DAISYExtractor()
for img_path in tqdm(
    QUERY_IMG_DIR.glob("*.jpg"),
    desc="Computing keypoints for the query set",
    total=len(list(QUERY_IMG_DIR.glob("*.jpg"))),
):
    image = cv2.imread(str(img_path))
    keypoints_sift, descriptors_sift = sift_extractor.compute_features(image)
    #keypoints_hog, descriptors_hog = hog_extractor.compute_features(image)
    keypoints_color_sift, descriptors_color_sift = color_sift_extractor.compute_features(image)
    keypoints_gloh, descriptors_gloh = gloh_extractor.compute_features(image)
    #keypoints_daisy, descriptors_daisy = daisy_extractor.compute_features(image)

    # Draw keypoints on the original image
    k1 = cv2.drawKeypoints(image, keypoints_sift, outImage=None,
                                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    k2 = cv2.drawKeypoints(image, keypoints_color_sift, outImage=None,
                                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    k3 = cv2.drawKeypoints(image, keypoints_gloh, outImage=None,
                                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #k4 = cv2.drawKeypoints(image, keypoints_hog, outImage=None,
    #                                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #k5 = cv2.drawKeypoints(image, keypoints_daisy, outImage=None,
    #                                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Show the image with keypoints
    image_with_keypoints = np.concatenate((k1, k2, k3), axis=1)
    cv2.imshow("Image with Keypoints: SIFT, Color SIFT, GLOH", image_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


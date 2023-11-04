from pathlib import Path

import numpy as np
import numpy as np
from descriptors import *
from distances import *
from tqdm import tqdm
from bg_removal import *
import os
from text_detection import *
from noise_removal import *
from similar_artist import *
from text_combination import *
from bg_removal import *

# set paths
QUERY_IMG_DIR = Path(os.path.join("data", "qsd1_w4"))
#QUERY_IMG_DIR = Path(os.path.join("..", "data", "Week4", "qsd1_w4"))

sift_extractor = SIFTExtractor()
color_sift_extractor = ColorSIFTExtractor()
gloh_extractor = GLOHExtractor()
orb_extractor = ORBExtractor()
kaze_extractor = KAZEExtractor()

bg_removal = RemoveBackgroundV3()
NOISE_FILTER = Median()
NAME_FILTER = Average()
TEXT_DETECTOR = TextDetection()
HAS_NOISE = SaltPepperNoise(noise_filter=NOISE_FILTER,
                                name_filter=NAME_FILTER,
                                text_detector=TEXT_DETECTOR)

i = 0
for img_path in tqdm(
    QUERY_IMG_DIR.glob("*.jpg"),
    desc="Computing keypoints for the query set",
    total=len(list(QUERY_IMG_DIR.glob("*.jpg"))),
):
    img = cv2.imread(str(img_path))
    denoised_image = HAS_NOISE(img) 
    imgs = bg_removal(denoised_image)
    if i > 3:
        break
    i += 1
    for image in imgs:
        keypoints_sift, descriptors_sift = sift_extractor(image)
        keypoints_color_sift, descriptors_color_sift = color_sift_extractor(image)
        keypoints_gloh, descriptors_gloh = gloh_extractor(image)
        keypoints_orb, descriptors_orb = orb_extractor(image)
        keypoints_kaze, descriptors_kaze = kaze_extractor(image)
        
        # Draw keypoints on the original image
        k1 = cv2.drawKeypoints(image, keypoints_sift, outImage=None,
                                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        k2 = cv2.drawKeypoints(image, keypoints_color_sift, outImage=None,
                                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        k3 = cv2.drawKeypoints(image, keypoints_gloh, outImage=None,
                                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        k4 = cv2.drawKeypoints(image, keypoints_orb, outImage=None,
                                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        k5 = cv2.drawKeypoints(image, keypoints_kaze, outImage=None,
                                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Show the image with keypoints
        image_with_keypoints = np.concatenate((k1, k2, k3, k4, k5), axis=1)
        cv2.imshow("Image with Keypoints: SIFT, Color SIFT, GLOH, ORB, KAZE", image_with_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #cv2.imwrite("results/keypoints/week4_sift_color_sift_gloh_orb_kaze_keypoints_"+str(i)+".jpg", image_with_keypoints)

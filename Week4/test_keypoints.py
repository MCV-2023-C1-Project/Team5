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
from distances import KeypointsMatcher

# set paths
QUERY_IMG_DIR = Path(os.path.join("data", "qsd1_w4"))
REF_IMG_DIR = Path(os.path.join("data", "BBDD"))

sift_extractor = SIFTExtractor()
color_sift_extractor = ColorSIFTExtractor()
gloh_extractor = GLOHExtractor()
orb_extractor = ORBExtractor()
kaze_extractor = KAZEExtractor()

bg_removal = RemoveBackgroundV3()
NOISE_FILTER = Median()
NAME_FILTER = Average()
TEXT_DETECTOR = TextDetectionV2()
HAS_NOISE = SaltPepperNoise(noise_filter=NOISE_FILTER,
                                name_filter=NAME_FILTER,
                                text_detector=TEXT_DETECTOR)

def matchers_example():
    q_img = cv2.imread(str(QUERY_IMG_DIR / "00005.jpg"))
    denoised_image = HAS_NOISE(q_img) 
    imgs = bg_removal(denoised_image)
    matcher = KeypointsMatcher(cv2.NORM_L2, 0.75)
    results = {}
    for image in imgs:
        text_mask = TEXT_DETECTOR.get_text_mask(image)
        
        _, descriptors_sift = sift_extractor(image, text_mask)
        # _, descriptors_color_sift = color_sift_extractor(image, text_mask)
        # _, descriptors_gloh = gloh_extractor(image, text_mask)
        # _, descriptors_orb = orb_extractor(image, text_mask)
        # _, descriptors_kaze = kaze_extractor(image, text_mask)
        
        for img_path in tqdm(
            REF_IMG_DIR.glob("*.jpg"),
            desc="Computing keypoints for the reference set",
            total=len(list(REF_IMG_DIR.glob("*.jpg"))),
        ):    
            ref_image = cv2.imread(str(img_path))
            _, ref_descriptors_sift = sift_extractor(ref_image)
            # _, ref_descriptors_color_sift = color_sift_extractor(ref_image)
            # _, ref_descriptors_gloh = gloh_extractor(ref_image)
            # _, ref_descriptors_orb = orb_extractor(ref_image)
            # _, ref_descriptors_kaze = kaze_extractor(ref_image)


            match = matcher(descriptors_sift, ref_descriptors_sift)
            results.update({int(img_path.stem[-5:]): match})
    
        sorted_dict = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
        print(sorted_dict)

def keypoints_example():
    for img_path in tqdm(
        QUERY_IMG_DIR.glob("*.jpg"),
        desc="Computing keypoints for the query set",
        total=len(list(QUERY_IMG_DIR.glob("*.jpg"))),
    ):
        img = cv2.imread(str(img_path))
        denoised_image = HAS_NOISE(img) 
        imgs = bg_removal(denoised_image)

        for image in imgs:
            keypoints_sift, _ = sift_extractor(image)
            keypoints_color_sift, _ = color_sift_extractor(image)
            keypoints_gloh, _ = gloh_extractor(image)
            keypoints_orb, _ = orb_extractor(image)
            keypoints_kaze, _ = kaze_extractor(image)
            
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
            
if __name__ == "__main__":
    # keypoints_example()
    matchers_example()
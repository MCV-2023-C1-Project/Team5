from pathlib import Path
from PIL import Image
import numpy as np
from descriptors import *
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
QUERY_IMG_DIR = Path(os.path.join("data", "qsd1_w4"))
REF_IMG_DIR = Path(os.path.join("data", "BBDD"))
RESULT_OUT_PATH = Path(os.path.join("results.pkl"))

# set hyper-parameters
DESCRIPTOR_FN = SIFTExtractor()
NOISE_FILTER = Median()
NAME_FILTER = Average()
TEXT_DETECTOR = TextDetectionV2()
HAS_NOISE = SaltPepperNoise(noise_filter=NOISE_FILTER,
                              name_filter=NAME_FILTER,
                              text_detector=TEXT_DETECTOR)
BG_REMOVAL_FN = RemoveBackgroundV3()

query_set = {}
for img_path in tqdm(
    QUERY_IMG_DIR.glob("*.jpg"),
    desc="Computing descriptors for the query set",
    total=len(list(QUERY_IMG_DIR.glob("*.jpg"))),
):
    idx = int(img_path.stem[-5:])
    img = Image.open(img_path)
    img = np.array(img)
    denoised_img = HAS_NOISE(img)
    imgs = BG_REMOVAL_FN(denoised_img)
    set_images = []
    for i, img in enumerate(imgs):
        text_mask = TEXT_DETECTOR.get_text_mask(img)
        _, descriptors = DESCRIPTOR_FN(img, text_mask)
        set_images.append(descriptors)

    query_set[idx] = set_images

ref_set = {}
for img_path in tqdm(
    REF_IMG_DIR.glob("*.jpg"),
    desc="Computing descriptors for the reference set",
    total=len(list(REF_IMG_DIR.glob("*.jpg"))),
):
    idx = int(img_path.stem[-5:])
    img = Image.open(img_path)
    img = np.array(img)
    _, descriptors = DESCRIPTOR_FN(img)
    ref_set[idx] = descriptors

result = []
for i in range(len(query_set)):
    q_list = []
    print("Evaluating image {}...".format(i))
    for query in query_set[i]:
        q_list.append(retrieve(query, ref_set))
    result.append(q_list)

print(result)

with open(RESULT_OUT_PATH, "wb") as file:
    pickle.dump(result, file)

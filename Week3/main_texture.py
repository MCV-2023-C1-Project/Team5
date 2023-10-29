from pathlib import Path
from PIL import Image
import numpy as np
from descriptors import *
from distances import *
from retrieval import retrieve
from tqdm import tqdm
from bg_removal import *
import pandas as pd
import os
from utils import mapk as mapk_v1, multi_mapk as mapk

# set paths
QUERY_IMG_DIR = Path(os.path.join("data", "qsd2_w3", "non_augmented"))
REF_IMG_DIR = Path(os.path.join("data", "BBDD"))
GT_RET = Path(os.path.join("data", "qsd2_w3", "gt_corresps.pkl"))
USE_V2 = True

gt = pd.read_pickle(GT_RET)

K = 10
SPLIT_SHAPE = (20, 20)
TEXTURE_DESCRIPTOR = SpatialDescriptor(DiscreteCosineTransform(), SPLIT_SHAPE)
# TEXTURE_DESCRIPTOR = SpatialDescriptor(LocalBinaryPattern(numPoints=24, radius=8), SPLIT_SHAPE)
DISTANCE_FN = Cosine()
REMOVE_BG = RemoveBackgroundV2()
TEXT_DETECTOR = TextDetection()

# generate descriptors for the query and for the reference datasets,
# store them as dictionaries {idx(int): descriptor(NumPy array)}
query_set = {}
for img_path in tqdm(
    QUERY_IMG_DIR.glob("*.jpg"),
    desc="Computing descriptors for the query set",
    total=len(list(QUERY_IMG_DIR.glob("*.jpg"))),
):
    idx = int(img_path.stem[-5:])
    img = Image.open(img_path)
    img = np.array(img)
    if USE_V2:
        imgs = REMOVE_BG(img)
        set_images = []
        for img in imgs:
            text_mask = TEXT_DETECTOR.get_text_mask(img)
            set_images.append(TEXTURE_DESCRIPTOR(img, text_mask)) 
        query_set[idx] = set_images
    else:
        query_set[idx] = TEXTURE_DESCRIPTOR(img) 

ref_set = {}
for img_path in tqdm(
    REF_IMG_DIR.glob("*.jpg"),
    desc="Computing descriptors for the reference set",
    total=len(list(REF_IMG_DIR.glob("*.jpg"))),
):
    idx = int(img_path.stem[-5:])
    img = Image.open(img_path)
    img = np.array(img)
    ref_set[idx] = TEXTURE_DESCRIPTOR(img)  # add "idx: descriptor" pair

if USE_V2:
    result = []
    for i in range(len(query_set)):
        q_list = []
        for query in query_set[i]:
            q_list.append(retrieve(query, ref_set, K, DISTANCE_FN))
        result.append(q_list)
else:
    result = []
    for i in range(len(query_set)):
        result.append(retrieve(query_set[i], ref_set, K, DISTANCE_FN))
    
# evaluate results
if USE_V2:
    metric = mapk(gt, result, k=5)
else:
    metric = mapk_v1(gt, result, k=1)

print(metric)


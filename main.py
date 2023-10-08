from pathlib import Path
from PIL import Image
import numpy as np
from descriptors import *
from distances import *
from retrieval import retrieve
import pickle
from tqdm import tqdm
from bg_removal import *
import os

# set paths
QUERY_IMG_DIR = Path(os.path.join("data", "Week1", "qst2_w1"))
REF_IMG_DIR = Path(os.path.join("data", "Week1", "BBDD"))
RESULT_OUT_PATH = Path("result.pkl")

# set hyper-parameters
DESCRIPTOR_FN = Histogram(color_model="yuv", bins=25, range=(0, 255))
K = 10
DISTANCE_FN = Intersection()
BG_REMOVAL_FN = RemoveBackground()

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
    img = BG_REMOVAL_FN(img)
    query_set[idx] = DESCRIPTOR_FN(img)  # add "idx: descriptor" pair

ref_set = {}
for img_path in tqdm(
    REF_IMG_DIR.glob("*.jpg"),
    desc="Computing descriptors for the reference set",
    total=len(list(REF_IMG_DIR.glob("*.jpg"))),
):
    idx = int(img_path.stem[-5:])
    img = Image.open(img_path)
    img = np.array(img)
    ref_set[idx] = DESCRIPTOR_FN(img)  # add "idx: descriptor" pair

# define queries nested list of indices (by default, whole query set)
queries = [[idx] for idx in range(len(query_set))]

# use retrieval api to obtain most similar to the queries samples
# from the reference dataset
result = [
    # access query with "[0]" since queries contain dummy list 'dimension'
    retrieve(query_set[query[0]], ref_set, K, DISTANCE_FN)
    for query in queries
]

# save resulting nested lists as pickle files
with open(RESULT_OUT_PATH, "wb") as file:
    pickle.dump(result, file)

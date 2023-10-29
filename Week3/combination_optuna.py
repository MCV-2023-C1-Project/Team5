from pathlib import Path
from PIL import Image
import numpy as np
from descriptors import *
from distances import *
from retrieval_combined import retrieve_combined
from tqdm import tqdm
from bg_removal import *
import optuna
import pandas as pd
import os


def apk(actual, predicted, k=10):
    """
    source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def compute_mapk(gt, hypo, k_val):
    """
    source: https://github.com/MCV-2023-C1-Project/mcv-c1-code/blob/main/score_painting_retrieval.py#L62
    """

    apk_list = []
    for ii, query in enumerate(gt):
        for jj, sq in enumerate(query):
            apk_val = 0.0
            if len(hypo[ii]) > jj:
                apk_val = apk([sq], hypo[ii][jj], k_val)
            apk_list.append(apk_val)

    return np.mean(apk_list)


# set paths
QUERY_IMG_DIR = Path(os.path.join("..", "data", "Week3", "qsd1_w3", "non_augmented"))
REF_IMG_DIR = Path(os.path.join("..", "data", "Week1", "BBDD"))
GT_RET = Path(os.path.join("..", "data", "Week3", "qsd1_w3", "gt_corresps.pkl"))

gt = pd.read_pickle(GT_RET)

SPLIT_SHAPE = (20, 20)
TEXTURE_DESCRIPTOR_1 = SpatialDescriptor(DiscreteCosineTransform(num_coeff=4), SPLIT_SHAPE)
COLOR_DESCRIPTOR_1 = SpatialDescriptor(Histogram(color_model="yuv", bins=25, range=(0, 255)), SPLIT_SHAPE)
DISTANCE_FN_TEXTURE = Cosine()
DISTANCE_FN_COLOR = Intersection()
REMOVE_BG = RemoveBackgroundV2()
TEXT_DETECTOR = TextDetection()
K = 10

query_set_texture = {}
query_set_color = {}

v2 = False

for img_path in tqdm(
        QUERY_IMG_DIR.glob("*.jpg"),
        desc="Computing descriptors for the query set",
        total=len(list(QUERY_IMG_DIR.glob("*.jpg"))),
):
    idx = int(img_path.stem[-5:])
    img = cv2.imread(str(img_path))
    imgs = REMOVE_BG(img)
    if v2:
        set_images_texture = []
        set_images_color = []
        for img in imgs:
            text_mask = TEXT_DETECTOR.get_text_mask(img)
            set_images_texture.append(TEXTURE_DESCRIPTOR_1(img, text_mask))  # add "idx: descriptor" pair
            set_images_color.append(COLOR_DESCRIPTOR_1(img, text_mask))  # add "idx: descriptor" pair
        query_set_texture[idx] = set_images_texture
        query_set_color[idx] = set_images_color
    else:
        text_mask = TEXT_DETECTOR.get_text_mask(img)
        query_set_texture[idx] = TEXTURE_DESCRIPTOR_1(img, text_mask)
        query_set_color[idx] = COLOR_DESCRIPTOR_1(img, text_mask)

ref_set_texture = {}
ref_set_color = {}
for img_path in tqdm(
        REF_IMG_DIR.glob("*.jpg"),
        desc="Computing descriptors for the reference set",
        total=len(list(REF_IMG_DIR.glob("*.jpg"))),
):
    idx = int(img_path.stem[-5:])
    img = cv2.imread(str(img_path))
    ref_set_texture[idx] = TEXTURE_DESCRIPTOR_1(img)
    ref_set_color[idx] = COLOR_DESCRIPTOR_1(img)


def objective(trial):
    K = 10
    index = trial.suggest_int("index", 0, 10)
    combinations = [(0,1),(0.1,0.9),(0.2,0.8),(0.3,0.7),(0.4,0.6),(0.5,0.5),
                    (0.6,0.4),(0.7,0.3),(0.8,0.2),(0.9,0.1),(1,0)]
    weight_1 = combinations[index][0]
    weight_2 = combinations[index][1]

    # generate descriptors for the query and for the reference datasets,
    # store them as dictionaries {idx(int): descriptor(NumPy array)}


    queries = [[idx] for idx in range(len(query_set_texture))]
    result = [
        retrieve_combined(query_set_texture[query[0]], ref_set_texture, DISTANCE_FN_TEXTURE, weight_1,
                          query_set_color[query[0]], ref_set_color, DISTANCE_FN_COLOR, weight_2, K)
        for query in queries
    ]

    # evaluate results
    metric = mapk(gt, result, k=10)
    return metric


search_space = {
    "index": [0,
              1,
              2,
              3,
              4,
              5,
              6,
              7,
              8,
              9,
              10],
}
study = optuna.create_study(
   sampler=optuna.samplers.GridSampler(search_space),
   direction="maximize",  # redundand, since grid search
   storage="sqlite:///hparam.db",
   study_name="v999999999999_idx",
)
study.optimize(objective)

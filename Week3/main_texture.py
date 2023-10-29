from pathlib import Path
from PIL import Image
import numpy as np
from descriptors import *
from distances import *
from retrieval import retrieve
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
QUERY_IMG_DIR = Path(os.path.join("data", "qsd1_w3", "non_augmented"))
REF_IMG_DIR = Path(os.path.join("data", "BBDD"))
GT_RET = Path(os.path.join("data", "qsd1_w3", "gt_corresps.pkl"))

gt = pd.read_pickle(GT_RET)


if __name__ == "__main__":
    SPLIT_SHAPE = (20, 20)
    TEXTURE_DESCRIPTOR_1 = SpatialDescriptor(DiscreteCosineTransform(num_coeff=4), SPLIT_SHAPE)
    DISTANCE_FN = Cosine()
    REMOVE_BG = RemoveBackgroundV2()
    TEXT_DETECTOR = TextDetection()
    K = 10

    query_set = {}
    
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
            set_images = []
            for img in imgs:
                text_mask = TEXT_DETECTOR.get_text_mask(img)
                set_images.append(TEXTURE_DESCRIPTOR_1(img, text_mask))  # add "idx: descriptor" pair
            query_set[idx] = set_images
        else:
            text_mask = TEXT_DETECTOR.get_text_mask(img)
            query_set[idx] = TEXTURE_DESCRIPTOR_1(img, text_mask)

    ref_set = {}
    for img_path in tqdm(
        REF_IMG_DIR.glob("*.jpg"),
        desc="Computing descriptors for the reference set",
        total=len(list(REF_IMG_DIR.glob("*.jpg"))),
    ):
        idx = int(img_path.stem[-5:])
        img = cv2.imread(str(img_path))
        ref_set[idx] = TEXTURE_DESCRIPTOR_1(img)


    if v2:
        result = []
        for i in range(len(query_set)):
            q_list = []
            for query in query_set[i]:
                q_list.append(retrieve(query, ref_set, K, DISTANCE_FN))
            result.append(q_list)
    else:
        queries = [[idx] for idx in range(len(query_set))]
        result = [
            retrieve(query_set[query[0]], ref_set, K, DISTANCE_FN)
            for query in queries
        ]

    # evaluate results
    metric = mapk(gt, result, k=10)
    print(metric)
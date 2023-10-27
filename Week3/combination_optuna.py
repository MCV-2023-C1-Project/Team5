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


def objective(trial):
    # set hyper-parameters
    DESCRIPTOR_FN = Histogram(color_model="hsv", bins=25, range=(0, 255))
    TEXTURE_DESCRIPTOR_1 = DiscreteCosineTransform()
    TEXTURE_DESCRIPTOR_2 = LocalBinaryPattern(numPoints=24, radius=8)
    K = 10
    INDEX = trial.suggest_int("index", 0, 2)
    TUPLES = [(DESCRIPTOR_FN, Intersection()), (TEXTURE_DESCRIPTOR_1, Euclidean()), (TEXTURE_DESCRIPTOR_2, Euclidean())]

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
        query_set[idx] = TUPLES[INDEX][0](img)  # add "idx: descriptor" pair

    ref_set = {}
    for img_path in tqdm(
        REF_IMG_DIR.glob("*.jpg"),
        desc="Computing descriptors for the reference set",
        total=len(list(REF_IMG_DIR.glob("*.jpg"))),
    ):
        idx = int(img_path.stem[-5:])
        img = Image.open(img_path)
        img = np.array(img)
        ref_set[idx] = TUPLES[INDEX][0](img)  # add "idx: descriptor" pair

    # define queries nested list of indices (by default, whole query set)
    queries = [[idx] for idx in range(len(query_set))]

    # use retrieval api to obtain most similar to the queries samples
    # from the reference dataset
    result = [
        # access query with "[0]" since queries contain dummy list 'dimension'
        retrieve(query_set[query[0]], ref_set, K, TUPLES[INDEX][1])
        for query in queries
    ]

    # evaluate results
    metric = mapk(gt, result, k=10)
    return metric


search_space = {
    "index": [0, # histogram with intersection
              1, # dct with euclidean
              2],# lbp with euclidean
}
study = optuna.create_study(
    sampler=optuna.samplers.GridSampler(search_space),
    direction="maximize",  # redundand, since grid search
    storage="sqlite:///hparam.db",
    study_name="v999_idx",
)
study.optimize(objective)

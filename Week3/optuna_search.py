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
QUERY_IMG_DIR = Path(os.path.join("data", "Week1", "qsd1_w1"))
REF_IMG_DIR = Path(os.path.join("data", "Week1", "BBDD"))
GT_RET = Path(os.path.join("data", "Week1", "qsd1_w1", "gt_corresps.pkl"))

gt = pd.read_pickle(GT_RET)


def objective(trial):
    NUM_BINS = trial.suggest_int("num_bins", 2, 256)
    COLOR_MODEL = trial.suggest_categorical("color_model", ["lab", "yuv"])

    # set hyper-parameters
    DESCRIPTOR_FN = Histogram(color_model=COLOR_MODEL, bins=NUM_BINS, range=(0, 255))
    K = 10
    DISTANCE_FN = trial.suggest_categorical(
        "distance_fn",
        [
            "Euclidean",
            "Correlation",
            "KullbackLeibler",
            "Cosine",
            "Intersection",
        ],
    )
    DISTANCE_FN = {
        "Euclidean": Euclidean,
        "Correlation": Correlation,
        "KullbackLeibler": KullbackLeibler,
        "Cosine": Cosine,
        "Intersection": Intersection,
    }[DISTANCE_FN]()
    BG_REMOVAL_FN = Identity()

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

    # evaluate results
    metric = mapk(gt, result, k=10)
    return metric


search_space = {
    "num_bins": [2, 7, 10, 15, 25, 50, 100, 150, 200, 256],
    "color_model": ["lab", "yuv"],
    "distance_fn": [
        "Euclidean",
        "Correlation",
        "KullbackLeibler",
        "Cosine",
        "Intersection",
    ],
}
study = optuna.create_study(
    sampler=optuna.samplers.GridSampler(search_space),
    direction="maximize",  # redundand, since grid search
    storage="sqlite:///hparam.db",
    study_name="v9_idx",
)
study.optimize(objective)

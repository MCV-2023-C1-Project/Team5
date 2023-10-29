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
from utils import mapk as mapk_v1

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
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

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

    result = []
    for a, p in zip(actual, predicted):
        if len(a) != len(p):
            result.append(0)
        else:
            for b, c in zip(a, p):
                    result.append(apk([b], c, k))
    mean = np.mean(result)
    return mean

def compute_mapk(gt,hypo,k_val):
    """
    source: https://github.com/MCV-2023-C1-Project/mcv-c1-code/blob/main/score_painting_retrieval.py#L62
    """

    apk_list = []
    for ii,query in enumerate(gt):
        for jj,sq in enumerate(query):
            apk_val = 0.0
            if len(hypo[ii]) > jj:
                apk_val = apk([sq],hypo[ii][jj], k_val)
            apk_list.append(apk_val)
            
    return np.mean(apk_list)


# set paths
QUERY_IMG_DIR = Path(os.path.join("data", "qsd2_w3", "non_augmented"))
REF_IMG_DIR = Path(os.path.join("data", "BBDD"))
GT_RET = Path(os.path.join("data", "qsd2_w3", "gt_corresps.pkl"))
USE_V2 = True

gt = pd.read_pickle(GT_RET)


def objective(trial):
    # set hyper-parameters
    SPLIT_SHAPE = (20, 20)
    TEXTURE_DESCRIPTOR_1 = SpatialDescriptor(DiscreteCosineTransform(), SPLIT_SHAPE)
    TEXTURE_DESCRIPTOR_2 = SpatialDescriptor(LocalBinaryPattern(numPoints=24, radius=8), SPLIT_SHAPE)
    K = 10
    INDEX = trial.suggest_int("index", 0, 5)
    TUPLES = [(TEXTURE_DESCRIPTOR_1, Cosine()), (TEXTURE_DESCRIPTOR_2, Cosine()), (TEXTURE_DESCRIPTOR_1, KullbackLeibler()), (TEXTURE_DESCRIPTOR_2, KullbackLeibler()),
              (TEXTURE_DESCRIPTOR_1, Bhattacharyya()), (TEXTURE_DESCRIPTOR_2, Bhattacharyya())]
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
                set_images.append(TUPLES[INDEX][0](img, text_mask)) 
            query_set[idx] = set_images
        else:
            query_set[idx] = TUPLES[INDEX][0](img) 

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

    if USE_V2:
        result = []
        for i in range(len(query_set)):
            q_list = []
            for query in query_set[i]:
                q_list.append(retrieve(query, ref_set, K, TUPLES[INDEX][1]))
            result.append(q_list)
    else:
        result = []
        for i in range(len(query_set)):
            result.append(retrieve(query_set[i], ref_set, K, TUPLES[INDEX][1]))
        
    # evaluate results
    if USE_V2:
        metric = mapk(gt, result, k=5)
    else:
        metric = mapk_v1(gt, result, k=1)

    return metric


search_space = {
    "index": [0,
              1,
              2,
              3,
              4,
              5],
}
study = optuna.create_study(
    sampler=optuna.samplers.GridSampler(search_space),
    direction="maximize",  # redundand, since grid search
    storage="sqlite:///hparam.db",
    study_name="v7",
)
study.optimize(objective)
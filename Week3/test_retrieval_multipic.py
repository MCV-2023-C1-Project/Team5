import pandas as pd
from pathlib import Path
import numpy as np
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
PRED_RET_PATH = Path("resultS20x20.pkl")
GT_RET_PATH = Path(r"C:\Users\krupa\Desktop\qsd2_w2\qsd2_w2\gt_corresps.pkl")

from itertools import chain

pred = pd.read_pickle(PRED_RET_PATH)
# flattened_list = [list(chain(*inner_list)) for inner_list in pred]
gt = pd.read_pickle(GT_RET_PATH)
K=1
print(f"mAP@:{K}", mapk(gt, pred, K))


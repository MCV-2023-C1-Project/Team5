import numpy as np

import cv2


def close(img):
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closing


def erode(img):
    h, w = img.shape
    hper = 0.007
    wper = 0.007
    hker = int(h * hper)
    wker = int(w * wper)

    kernel = np.ones((hker, wker), np.uint8)
    erode = cv2.erode(img, kernel, iterations=5)
    
    return erode


def dilate(img):
    h, w = img.shape
    hper = 0.005
    wper = 0.005
    hker = int(h * hper)
    wker = int(w * wper)

    kernel = np.ones((hker, wker), np.uint8)
    dilate = cv2.dilate(img, kernel, iterations=5)

    return dilate


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
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

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

def multi_mapk(actual, predicted, k=10):
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
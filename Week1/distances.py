import numpy as np
import cv2
from scipy.spatial.distance import cosine


class Euclidean:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, a, b):
        result = np.sum((a - b) ** 2)
        return result


class Hellinger:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, a, b):
        result = np.sum(np.sqrt(a * b))
        result *= -1  # similarity to distance
        return result


class L1:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, a, b):
        result = np.abs(a - b).sum()
        return result


class Correlation:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, a, b):
        result = cv2.compareHist(
            a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_CORREL
        )
        result *= -1  # similarity to distance
        return result


class ChiSquare:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, a, b):
        result = cv2.compareHist(
            a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_CHISQR
        )
        return result


class KullbackLeibler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, a, b):
        result = cv2.compareHist(
            a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_KL_DIV
        )
        return result


class Bhattacharyya:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, a, b):
        result = cv2.compareHist(
            a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA
        )
        return result


class Intersection:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, a, b):
        result = cv2.compareHist(
            a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_INTERSECT
        )
        result *= -1  # similarity to distance
        return result


class Cosine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, a, b):
        result = cosine(a, b)  # cosine distance, no need in *(-1)
        return result

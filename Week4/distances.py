import numpy as np
import cv2
from scipy.spatial.distance import cosine
import editdistance


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


class Edit:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, a, b):
        if type(a) == str and type(b) == str:
            result = editdistance.eval(a, b)
        else:
            result = np.inf
        return result


class KeypointsMatcher:
    def __init__(self, distance: any, threshold: float, **kwargs):
        # By default, `distance` is cv.NORM_L2. It is good for SIFT, SURF etc (cv.NORM_L1 is also there). 
        # For binary string based descriptors like ORB, BRIEF, BRISK etc, cv.NORM_HAMMING should be used, 
        # which used Hamming distance as measurement. If ORB is using WTA_K == 3 or 4, cv.NORM_HAMMING2 should be used.
        self.matcher = cv2.BFMatcher(distance, crossCheck=False)
        self.threshold = threshold
        self.kwargs = kwargs

    def get_matches(self, a: np.ndarray, b: np.ndarray) -> list:
        if a is None or b is None:
            return []
        
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        matches = self.matcher.match(a, b)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def __call__(self, a: np.ndarray, b: np.ndarray) -> int:
        return len(self.get_matches(a, b))

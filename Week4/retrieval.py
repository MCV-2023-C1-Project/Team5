from distances import KeypointsMatcher
import cv2
import numpy as np

MATCHER = KeypointsMatcher(cv2.NORM_L2, 0.75)
THRESHOLD = 5000

def retrieve(query_descriptor: np.ndarray, ref_set: dict) -> int:
    for ref_idx, ref_descriptor in ref_set.items():
        match = MATCHER(query_descriptor, ref_descriptor)
        
        if match > THRESHOLD:
            return ref_idx
 
    return -1

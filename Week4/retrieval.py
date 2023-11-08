from distances import KeypointsMatcher
import cv2
import numpy as np

MATCHER = KeypointsMatcher(cv2.NORM_L2, 0.75)
THRESHOLD = 2000

def retrieve(query_descriptor, ref_set, k, distance_function):
    distances = [
        distance_function(query_descriptor, ref_descriptor)
        for ref_descriptor in ref_set.values()
    ]
    
    distance_ranks = np.argsort(distances)
    sorted_image_indices = np.array(list(ref_set.keys()))[distance_ranks]
    result = sorted_image_indices[:k].tolist()
    return result

def match(query_descriptor: np.ndarray, ref_set: dict, previous_result: list) -> int:
    descriptors = {idx: ref_set[idx] for idx in previous_result}
    matches = {key: MATCHER(query_descriptor, descriptor) for key, descriptor in descriptors.items()}        
    matches = dict(sorted(matches.items(), key=lambda x: x[1], reverse=True))
    
    for idx, match in matches.items():        
        if match > THRESHOLD:
            return int(idx)
 
    return -1

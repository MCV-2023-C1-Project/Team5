from distances import KeypointsMatcher
import cv2
import numpy as np

MATCHER = KeypointsMatcher(cv2.NORM_L2, 0.75)
THRESHOLD = 0.075

def retrieve(query_descriptor, ref_set, k, distance_function):
    distances = [
        distance_function(query_descriptor, ref_descriptor)
        for ref_descriptor in ref_set.values()
    ]
    
    distance_ranks = np.argsort(distances)
    sorted_image_indices = np.array(list(ref_set.keys()))[distance_ranks]
    result = sorted_image_indices[:k].tolist()
    return result

def match(query_descriptor: np.ndarray, ref_set: dict, previous_result: list, k: int = 10) -> int:
    descriptors = {idx: ref_set[idx] for idx in previous_result}
    matches = {key: MATCHER(query_descriptor, descriptor) for key, descriptor in descriptors.items()}        
    matches = dict(sorted(matches.items(), key=lambda x: x[1], reverse=True))
    
    if list(matches.values())[0] > query_descriptor.shape[0] * THRESHOLD:
        return list(matches.keys())[:k+1]
 
    return [-1]

import numpy as np
from distances import *
DISTANCE_1 = Intersection()
DISTANCE_2 = Euclidean()
def retrieve_combined(query_descriptor_1, ref_set_1, distance_function_1, weight_1,
                      query_descriptor_2, ref_set_2, distance_function_2, weight_2, k):
    """Return k most similar images for each image in queries.

    Args:
        query_descriptor_1: NumPy array
        ref_set_1: Dictionary, with keys as reference set image indices
        (integers), and values as their descriptors (NumPy arrays).
        distance_function_1: Function used to compute distance between
            image descriptors.
        weight_1: weight of the first descriptor
        query_descriptor_2: NumPy array
        ref_set_2: Dictionary, with keys as reference set image indices
        (integers), and values as their descriptors (NumPy arrays).
        k: Integer, number of most similar images returned for each
        query image.
        distance_function_2: Function used to compute distance between
            image descriptors.
        weight_2: weight of the second descriptor


    Returns:
        Nested list, each element is a list, containing k
        integers, representing indices of database images, that are most
        similar to corresponding query.

    """

    # compute distances of given descriptor to each descriptor in the ref set
    distances_1 = [
        DISTANCE_1(query_descriptor_1, ref_descriptor)
        for ref_descriptor in ref_set_1.values()
    ]
    distances_2 = [
        DISTANCE_2(query_descriptor_2, ref_descriptor)
        for ref_descriptor in ref_set_2.values()
    ]
    distances = np.array(distances_1) * weight_1 + np.array(distances_2) * weight_2
    # obtain array of indices that would sort distances
    distance_ranks = np.argsort(distances)
    # sort image indices (names) wrt distance
    sorted_image_indices = np.array(list(ref_set_1.keys()))[distance_ranks] #could also be ref_set_2.keys()
    # retain only k closest images
    result = sorted_image_indices[:k].tolist()

    return result

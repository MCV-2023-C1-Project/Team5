import numpy as np


def retrieve(query_descriptor, ref_set, k, distance_function):
    """Return k most similar images for each image in queries.

    Args:
        query_descriptor: NumPy array
        ref_set: Dictionary, with keys as reference set image indices
        (integers), and values as their descriptors (NumPy arrays).
        k: Integer, number of most similar images returned for each
        query image.
        distance_function: Function used to compute distance between
            image descriptors.

    Returns:
        Nested list, each element is a list, containing k
        integers, representing indices of database images, that are most
        similar to corresponding query.

    """

    # compute distances of given descriptor to each descriptor in the ref set
    distances = [
        distance_function(query_descriptor, ref_descriptor)
        for ref_descriptor in ref_set.values()
    ]
    # obtain array of indices that would sort distances
    distance_ranks = np.argsort(distances)
    # sort image indices (names) wrt distance
    sorted_image_indices = np.array(list(ref_set.keys()))[distance_ranks]
    # retain only k closest images
    result = sorted_image_indices[:k].tolist()

    return result

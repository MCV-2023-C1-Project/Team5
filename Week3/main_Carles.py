from pathlib import Path
from PIL import Image
import numpy as np
from descriptors import *
from distances import *
from retrieval import retrieve
import pickle
from tqdm import tqdm
from bg_removal import *
import os
from text_detection import *
from noise_removal import *
from similar_artist import *
from combination import *

# set paths
QUERY_IMG_DIR = Path(os.path.join("data", "Week3", "qsd2_w3"))
REF_IMG_DIR = Path(os.path.join("data", "Week1", "BBDD"))
RESULT_OUT_PATH = Path("compared_result_qsd2_K10.pkl")

# # redefining paths for my machine, feel free to remove this block
# QUERY_IMG_DIR = Path(r"C:\Users\krupa\Desktop\qsd2_w2\qsd2_w2")
# REF_IMG_DIR = Path(r"D:\C1-Project (LEGACY)\data\BBDD")
# RESULT_OUT_PATH = Path("result3.pkl")

# set hyper-parameters
BASE_DESCRIPTOR = Histogram(color_model="yuv", bins=25, range=(0, 255))
SPLIT_SHAPE = (20, 20)  # (1, 1) is the same as not making spatial at all
DESCRIPTOR_FN = SpatialDescriptor(BASE_DESCRIPTOR, SPLIT_SHAPE)
K = 10
DISTANCE_FN = Intersection()
NOISE_FILTER = Median()
NAME_FILTER = Average()
TEXT_DETECTOR = TextDetection()
HAS_NOISE = Salt_Pepper_Noise(noise_filter=NOISE_FILTER,
                              name_filter=NAME_FILTER,
                              text_detector=TEXT_DETECTOR)

v2 = False
if QUERY_IMG_DIR.stem[-4:] == "2_w2":
    v2 = True
    BG_REMOVAL_FN = RemoveBackgroundV2()
else:
    BG_REMOVAL_FN = RemoveBackground()

path_query_csv = Path(os.path.join("data", "Week3", "paintings_db_w2d1.csv"))
Similar_Artist = SimilarArtist(path_query_csv=path_query_csv)

# generate descriptors for the query and for the reference datasets,
# store them as dictionaries {idx(int): descriptor(NumPy array)}
query_set = {}
query_artist = {}
for img_path in tqdm(
    QUERY_IMG_DIR.glob("*.jpg"),
    desc="Computing descriptors for the query set",
    total=len(list(QUERY_IMG_DIR.glob("*.jpg"))),
):
    idx = int(img_path.stem[-5:])
    img = Image.open(img_path)
    img = np.array(img)
    # Remove noise
    denoised_img = HAS_NOISE(img)
    #Image.fromarray(denoised_img).save("denoised/qsd2/{}.png".format(idx))
    # NOTE: text should be detected AFTER bg removal
    imgs = BG_REMOVAL_FN(denoised_img)

    if v2:
        set_images = []
        artists = []
        for img in imgs:
            text_mask = TEXT_DETECTOR.get_text_mask(img)
            text_coords = TEXT_DETECTOR.detect_text(img)
            set_images.append(DESCRIPTOR_FN(img, text_mask))  # add "idx: descriptor" pair
            artists.append(Similar_Artist(img, text_coords))
        query_set[idx] = set_images
        query_artist[idx] = artists
    else:
        text_mask = TEXT_DETECTOR.get_text_mask(imgs)
        text_coords = TEXT_DETECTOR.detect_text(imgs)
        artist = SimilarArtist(img, text_coords)
        query_set[idx] = DESCRIPTOR_FN(imgs, text_mask)
        query_artist[idx] = artist

ref_set = {}
for img_path in tqdm(
    REF_IMG_DIR.glob("*.jpg"),
    desc="Computing descriptors for the reference set",
    total=len(list(REF_IMG_DIR.glob("*.jpg"))),
):
    idx = int(img_path.stem[-5:])
    img = Image.open(img_path)
    img = np.array(img)
    ref_set[idx] = DESCRIPTOR_FN(img)  # add "idx: descriptor" pair

path_csv_bbdd = Path(os.path.join("data", "Week3", "paintings_db_bbdd.csv"))
Compare_Artist = CompareArtist(path_csv_bbdd=path_csv_bbdd)

if v2:
    """
        Retrieval without Text
    """
    # result = []
    # use retrieval api to obtain most similar to the queries samples
    # from the reference dataset

    # for idx in query_set.keys():
    #     q_list = []
    #     for query in query_set[idx]:
    #         q_list.append(retrieve(query, ref_set, K, DISTANCE_FN))
    #     result.append(q_list)

    """
        Retrieval with Text
    """
    result_dict = {}
    for idx in query_set.keys():
        q_list = []
        for query in query_artist[idx]:
            q_list.append(retrieve(query, ref_set, K, DISTANCE_FN))
        result_dict[idx] = q_list


    result = Compare_Artist.compare_multiple_paintings_artist(result_dict, query_artist)

else:
    """
        Retrieval without Text
    """
    # define queries nested list of indices (by default, whole query set)
    # queries = [[idx] for idx in range(len(query_set))]
    # # use retrieval api to obtain most similar to the queries samples
    # # from the reference dataset
    #
    # result = [
    #     # access query with "[0]" since queries contain dummy list 'dimension'
    #     retrieve(query_set[idx], ref_set, K, DISTANCE_FN)
    #     for idx in query_set.keys()
    # ]

    """
        Retrieval with Text
    """
    result_dict = {}
    for idx in query_set.keys():
        result_dict[idx] = retrieve(query_set[idx], ref_set, K, DISTANCE_FN)

    result = Compare_Artist.compare_artist(result_dict, query_artist)

# save resulting nested lists as pickle files
with open(RESULT_OUT_PATH, "wb") as file:
    pickle.dump(result, file)

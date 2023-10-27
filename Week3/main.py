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
from text_combination import *

# set paths
QUERY_IMG_DIR = Path(os.path.join("data", "Week3", "qsd1_w3"))
REF_IMG_DIR = Path(os.path.join("data", "Week1", "BBDD"))
RESULT_OUT_PATH = Path(os.path.join("results", "color_text_qsd1_K10.pkl"))

with_text_combination = True

# set hyper-parameters
BASE_DESCRIPTOR = Histogram(color_model="yuv", bins=25, range=(0, 255))
# BASE_DESCRIPTOR = LocalBinaryPattern(numPoints=8, radius=1)
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

path_csv_bbdd = Path(os.path.join("data", "paintings_db_bbdd.csv"))
path_txt_artists = Path(os.path.join(QUERY_IMG_DIR, "artists"))

Similar_Artist = ArtistReader(TEXT_DETECTOR,
                              path_query_csv=path_csv_bbdd,
                              save_txt_path=path_txt_artists)

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
    # Image.fromarray(denoised_img).save("denoised/qsd2/{}.png".format(idx))
    # NOTE: text should be detected AFTER bg removal
    if v2:
        imgs = BG_REMOVAL_FN(denoised_img)
        set_images = []
        artists = []
        for img in imgs:
            text_mask = TEXT_DETECTOR.get_text_mask(img)
            set_images.append(DESCRIPTOR_FN(img, text_mask))  # add "idx: descriptor" pair
            artist = Similar_Artist(img)
            Similar_Artist.save_txt(artist, idx)
            artists.append(artist)
        query_set[idx] = set_images
        query_artist[idx] = artists
    else:
        text_mask = TEXT_DETECTOR.get_text_mask(denoised_img)
        # text_coords = TEXT_DETECTOR.detect_text(denoised_img)
        artist = Similar_Artist(denoised_img)
        Similar_Artist.save_txt(artist, idx)
        # For color hist
        query_set[idx] = DESCRIPTOR_FN(denoised_img, text_mask)
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
        Retrieval with Text
    """
    if with_text_combination:
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
        result = []
        # use retrieval api to obtain most similar to the queries samples
        # from the reference dataset

        for idx in query_set.keys():
            q_list = []
            for query in query_set[idx]:
                q_list.append(retrieve(query, ref_set, K, DISTANCE_FN))
            result.append(q_list)

else:
    """
        Retrieval with Text
    """
    if with_text_combination:
        result_dict = {}
        for idx in query_set.keys():
            result_dict[idx] = retrieve(query_set[idx], ref_set, K, DISTANCE_FN)

        result = Compare_Artist.compare_artist(result_dict, query_artist)
    else:
        # define queries nested list of indices (by default, whole query set)
        queries = [[idx] for idx in range(len(query_set))]
        # use retrieval api to obtain most similar to the queries samples
        # from the reference dataset

        result = [
            # access query with "[0]" since queries contain dummy list 'dimension'
            retrieve(query_set[idx], ref_set, K, DISTANCE_FN)
            for idx in query_set.keys()
         ]


# save resulting nested lists as pickle files
with open(RESULT_OUT_PATH, "wb") as file:
    pickle.dump(result, file)

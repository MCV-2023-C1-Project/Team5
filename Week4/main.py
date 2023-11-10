from pathlib import Path
from PIL import Image
import numpy as np
from descriptors import *
from distances import *
from retrieval import retrieve, match
import pickle
from tqdm import tqdm
from bg_removal import *
import os
from text_detection import *
from noise_removal import *
from similar_artist import *
from text_combination import *

# set paths
QUERY_IMG_DIR = Path(os.path.join("data", "qsd1_w4"))
REF_IMG_DIR = Path(os.path.join("data", "BBDD"))
RESULT_OUT_PATH = Path(os.path.join("results.pkl"))

"""
    COLOR HISTOGRAM
"""
BASE_DESCRIPTOR = Histogram(color_model="yuv", bins=25, range=(0, 255))
SPLIT_SHAPE = (20, 20)
DESCRIPTOR_FN = SpatialDescriptor(BASE_DESCRIPTOR, SPLIT_SHAPE)
K = 10
DISTANCE_FN = Intersection()
KEPOINTS_FN = SIFTExtractor()
NOISE_FILTER = Median()
NAME_FILTER = Average()
TEXT_DETECTOR = TextDetectionV2()
HAS_NOISE = SaltPepperNoise(noise_filter=NOISE_FILTER,
                              name_filter=NAME_FILTER,
                              text_detector=TEXT_DETECTOR)
BG_REMOVAL_FN = RemoveBackgroundV3()

path_csv_bbdd = Path("paintings_db_bbdd.csv")
path_txt_artists = Path(os.path.join(QUERY_IMG_DIR, "artists"))

COMPARE_ARTIST = CompareArtist(path_csv_bbdd=path_csv_bbdd)
SIMILAR_ARTIST = ArtistReader(TEXT_DETECTOR,
                              path_bbdd_csv=path_csv_bbdd,
                              save_txt_path=path_txt_artists)

"""
    COLOR DESCRIPTORS
"""
query_set = {}
query_artist = {}
query_set_keypoints = {}
for img_path in tqdm(
    QUERY_IMG_DIR.glob("*.jpg"),
    desc="Computing descriptors for the query set",
    total=len(list(QUERY_IMG_DIR.glob("*.jpg"))),
):
    idx = int(img_path.stem[-5:])
    img = Image.open(img_path)
    img = np.array(img)
    # Remove noise

    denoised_path = Path(os.path.join("denoised", "qsd1_w4"))
    os.makedirs(denoised_path, exist_ok=True)
    # NOTE: text should be detected AFTER bg removal
    os.makedirs(path_txt_artists, exist_ok=True)
    file_name = os.path.join(path_txt_artists, f"{idx:05d}.txt")
    edges = cv2.Canny(img, 10, 80)
    denoised_img = HAS_NOISE(img)
    with open(file_name, 'w'):
        pass
    imgs = BG_REMOVAL_FN(denoised_img)
    set_images = []
    set_images_keypoints = []
    artists = []
    for img in imgs:
        text_mask = TEXT_DETECTOR(img)
        set_images.append(DESCRIPTOR_FN(img, text_mask))
        _, keypoints_descriptor = KEPOINTS_FN(img, text_mask)
        set_images_keypoints.append(keypoints_descriptor)
        artist = SIMILAR_ARTIST(img)
        SIMILAR_ARTIST.save_txt(artist, file_name)
        artists.append(artist)
    query_set[idx] = set_images
    query_artist[idx] = artists
    query_set_keypoints[idx] = set_images_keypoints

ref_set = {}
ref_set_keypoints = {}
for img_path in tqdm(
    REF_IMG_DIR.glob("*.jpg"),
    desc="Computing descriptors for the reference set",
    total=len(list(REF_IMG_DIR.glob("*.jpg"))),
):
    idx = int(img_path.stem[-5:])
    img = Image.open(img_path)
    img = np.array(img)
    ref_set[idx] = DESCRIPTOR_FN(img) 
    _, keypoints_descriptor = KEPOINTS_FN(img)
    ref_set_keypoints[idx] = keypoints_descriptor

"""
    GET RETRIEVAL FOR COLOR + TEXT
"""
with_text_combination = True
"""
        Retrieval with Text
"""
if with_text_combination:
    result_dict = {}
    for idx in range(len(query_set)):
        q_list = []
        for query in query_set[idx]:
            q_list.append(retrieve(query, ref_set, K, DISTANCE_FN))
        result_dict[idx] = q_list

    result = COMPARE_ARTIST.compare_multiple_paintings_artist(result_dict, query_artist)
else:
    """
            Retrieval without Text
    """
    result = []
    # use retrieval api to obtain most similar to the queries samples
    # from the reference dataset
    for idx in range(len(query_set)):
        q_list = []
        for query in query_set[idx]:
            q_list.append(retrieve(query, ref_set, K, DISTANCE_FN))
        result.append(q_list)

# Apply Keypoints to Result list.

"""
    KEYPOINTS
"""
pickle.dump(result, open("result.pkl", "wb"))
pickle.dump(query_set_keypoints, open("query_set_keypoints.pkl", "wb"))
pickle.dump(ref_set_keypoints, open("ref_set_keypoints.pkl", "wb"))

# result = pickle.load(open("result.pkl", "rb"))
# query_set_keypoints = pickle.load(open("query_set_keypoints.pkl", "rb"))
# ref_set_keypoints = pickle.load(open("ref_set_keypoints.pkl", "rb"))

result_keypoints = []    
for i in tqdm(range(len(query_set_keypoints)), desc="Matching keypoints for query images"):
    q_list = []
    for j, query in enumerate(query_set_keypoints[i]):
        q_list.append(match(query, ref_set_keypoints, result[i][j]))
    result_keypoints.append(q_list)

print(result_keypoints)

with open(RESULT_OUT_PATH, "wb") as file:
    pickle.dump(result_keypoints, file)

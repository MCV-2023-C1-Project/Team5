from pathlib import Path
from PIL import Image
import numpy as np

from retrieval_combined import retrieve_combined
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
QUERY_IMG_DIR = Path(os.path.join("..","data", "Week3", "qsd2_w3"))
REF_IMG_DIR = Path(os.path.join("..","data", "Week1", "BBDD"))
RESULT_OUT_PATH = Path(os.path.join("results", "all_qsd2_K5.pkl"))

with_text_combination = True

# set hyper-parameters
# BASE_DESCRIPTOR = Histogram(color_model="yuv", bins=25, range=(0, 255))
SPLIT_SHAPE = (20, 20)  # (1, 1) is the same as not making spatial at all
TEXTURE_DESCRIPTOR_1 = SpatialDescriptor(DiscreteCosineTransform(num_coeff=4), SPLIT_SHAPE)
COLOR_DESCRIPTOR_1 = SpatialDescriptor(Histogram(color_model="yuv", bins=25, range=(0, 255)), SPLIT_SHAPE)
K = 10
DISTANCE_FN_TEXTURE = Cosine()
DISTANCE_FN_COLOR = Intersection()
NOISE_FILTER = Median()
NAME_FILTER = Average()
TEXT_DETECTOR = TextDetection()
HAS_NOISE = Salt_Pepper_Noise(noise_filter=NOISE_FILTER,
                              name_filter=NAME_FILTER,
                              text_detector=TEXT_DETECTOR)

v2 = False
if QUERY_IMG_DIR.stem[-4:] == "2_w3":
    v2 = True
    BG_REMOVAL_FN = RemoveBackgroundV2()
else:
    BG_REMOVAL_FN = RemoveBackground()

path_csv_bbdd = Path("paintings_db_bbdd.csv")
path_txt_artists = Path(os.path.join(QUERY_IMG_DIR, "artists"))

Similar_Artist = ArtistReader(TEXT_DETECTOR,
                              path_bbdd_csv=path_csv_bbdd,
                              save_txt_path=path_txt_artists)

# generate descriptors for the query and for the reference datasets,
# store them as dictionaries {idx(int): descriptor(NumPy array)}
query_set_color = {}
query_set_texture = {}
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

    denoised_path = Path(os.path.join("denoised", "qsd1"))
    os.makedirs(denoised_path, exist_ok=True)
    # NOTE: text should be detected AFTER bg removal
    os.makedirs(path_txt_artists, exist_ok=True)
    file_name = os.path.join(path_txt_artists, f"{idx:05d}.txt")
    edges = cv2.Canny(img, 10, 80)
    denoised_img = HAS_NOISE(img)
    with open(file_name, 'w'):
        pass
    if v2:
        imgs = BG_REMOVAL_FN(denoised_img)
        set_images_color = []
        set_images_texture = []
        artists = []
        for i, img in enumerate(imgs):
            Image.fromarray(img).save(Path(os.path.join(denoised_path, "{}_{}.png".format(idx, i))))
            text_mask = TEXT_DETECTOR.get_text_mask(img)
            set_images_color.append(COLOR_DESCRIPTOR_1(img, text_mask))  # add "idx: descriptor" pair
            set_images_texture.append(TEXTURE_DESCRIPTOR_1(img, text_mask))  # add "idx: descriptor" pair
            artist = Similar_Artist(img)
            Similar_Artist.save_txt(artist, file_name)
            artists.append(artist)
        query_set_color[idx] = set_images_color
        query_set_texture[idx] = set_images_texture
        query_artist[idx] = artists

    else:
        text_mask = TEXT_DETECTOR.get_text_mask(denoised_img)
        # text_coords = TEXT_DETECTOR.detect_text(denoised_img)
        artist = Similar_Artist(denoised_img)
        Similar_Artist.save_txt(artist, file_name)
        # For color hist
        query_set_color[idx] = COLOR_DESCRIPTOR_1(denoised_img, text_mask)
        query_set_texture[idx] = TEXTURE_DESCRIPTOR_1(denoised_img, text_mask)
        query_artist[idx] = artist

ref_set_color = {}
ref_set_texture = {}
for img_path in tqdm(
    REF_IMG_DIR.glob("*.jpg"),
    desc="Computing descriptors for the reference set",
    total=len(list(REF_IMG_DIR.glob("*.jpg"))),
):
    idx = int(img_path.stem[-5:])
    img = Image.open(img_path)
    img = np.array(img)
    ref_set_color[idx] = COLOR_DESCRIPTOR_1(img)  # add "idx: descriptor" pair
    ref_set_texture[idx] = TEXTURE_DESCRIPTOR_1(img)  # add "idx: descriptor" pair


Compare_Artist = CompareArtist(path_csv_bbdd=path_csv_bbdd)
weight_1 = 0.7
weight_2 = 0.3
if v2:
    """
        Retrieval with Text
    """
    if with_text_combination:
        result_dict = {}
        for idx in range(len(query_set_texture)):
            q_list = []
            for i, query in enumerate(query_set_texture[idx]):
                q_list.append(
                    retrieve_combined(
                        query,
                        ref_set_texture,
                        DISTANCE_FN_TEXTURE,
                        weight_1,
                        query_set_color[idx][i],
                        ref_set_color,
                        DISTANCE_FN_COLOR,
                        weight_2,
                        K,
                    )
                )
            result_dict[idx] = q_list

        result = Compare_Artist.compare_multiple_paintings_artist(result_dict, query_artist)
    else:
        """
                Retrieval without Text
        """
        result = []
        # use retrieval api to obtain most similar to the queries samples
        # from the reference dataset

        for idx in range(len(query_set_texture)):
            q_list = []
            for query in query_set_texture[idx]:
                q_list.append(retrieve_combined(query_set_texture[query], ref_set_texture, DISTANCE_FN_TEXTURE, weight_1,
                          query_set_color[query], ref_set_color, DISTANCE_FN_COLOR, weight_2, K))
            result.append(q_list)

else:
    """
        Retrieval with Text
    """
    # if with_text_combination:
    #     result_dict = {}
    #     for idx in range(len(query_set)):
    #         result_dict[idx] = retrieve_combined(query_set[idx], ref_set, K, DISTANCE_FN)
    #
    #     result = Compare_Artist.compare_artist(result_dict, query_artist)
    # else:
    #     # define queries nested list of indices (by default, whole query set)
    #     queries = [[idx] for idx in range(len(query_set))]
    #     # use retrieval api to obtain most similar to the queries samples
    #     # from the reference dataset
    #
    #     result = [
    #         # access query with "[0]" since queries contain dummy list 'dimension'
    #         retrieve_combined(query_set[idx], ref_set, K, DISTANCE_FN)
    #         for idx in range(len(query_set))
    #      ]


# save resulting nested lists as pickle files
with open(RESULT_OUT_PATH, "wb") as file:
    pickle.dump(result, file)

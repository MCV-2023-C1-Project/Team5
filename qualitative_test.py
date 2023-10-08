import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

PRED_RET_PATH = Path("result.pkl")
REF_IMG_DIR = Path(os.path.join("data", "BBDD"))
QUERY_IMG_DIR = Path(os.path.join("data", "qsd1_w1"))

pred = pd.read_pickle(PRED_RET_PATH)

query_set_len = 30
queries = [[idx] for idx in range(query_set_len)]

K_TO_SHOW = 5

for query in queries:
    query_idx = query[0]
    query_img_path = (QUERY_IMG_DIR / f"{query_idx:0{5}d}").with_suffix(".jpg")
    query_img = Image.open(query_img_path)
    print(f"Query index: {query_idx}")
    plt.imshow(query_img)
    plt.show()

    for k, ret_idx in enumerate(pred[query_idx]):
        if k == K_TO_SHOW:
            break
        ref_img_path = (REF_IMG_DIR /("bbdd_" + f"{ret_idx:0{5}d}"))\
            .with_suffix(".jpg")
        ref_img = Image.open(ref_img_path)
        print(f"k: {k}")
        plt.imshow(ref_img)
        plt.show()

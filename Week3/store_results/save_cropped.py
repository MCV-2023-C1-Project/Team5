from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from bg_removal import *
import os

# set paths
IMG_DIR = Path(os.path.join("data", "Week2", "qsd2_w2"))
MASKS_OUT_DIR = Path("cropped_images")
PREPROCESS_FN = RemoveBackgroundV2()
os.makedirs(MASKS_OUT_DIR, exist_ok=True)

for img_path in tqdm(
    IMG_DIR.glob("*.jpg"), desc="Cropping Masks", total=len(list(IMG_DIR.glob("*.jpg")))
):
    img = Image.open(img_path)
    img = np.array(img)
    images = PREPROCESS_FN(img)

    for i, img in enumerate(images):
        path = img_path.stem + "_{}".format(i)
        Image.fromarray(img).save((MASKS_OUT_DIR / path).with_suffix(".png"))

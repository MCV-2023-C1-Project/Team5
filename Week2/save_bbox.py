import glob
import os

import numpy as np
from PIL import Image
import pickle

from text_detection import TextDetection
from bg_removal import RemoveBackgroundV2
from pathlib import Path

def split_name(file: str) -> str:
    return file.replace("\\", "/").split("/")[-1].split(".")[0]


def get_image_name(file: str) -> int:
    return int(split_name(file))


def transform_bbox(bbox: list) -> list:
    if bbox is None:
        return None

    x, y, w, h = bbox
    tlx = x
    tly = y
    brx = x + w
    bry = y + h
    return [tlx, tly, brx, bry]


def transform_gt_bbox(bbox: list) -> list:
    return np.array([bbox[0], bbox[2]]).flatten()


def main():
    path = Path(os.path.join("data", "Week2", "qst2_w2"))
    v2 = path.stem[-4:] == "2_w2"
    
    images = {
        get_image_name(file): Image.open(file)
        for file in glob.glob(os.path.join(path, "*.jpg"))
    }

    bboxes = []
    rm_text = TextDetection()
    all_bboxes = []
    for i in range(len(images)):
        print("Image", i)
        img = np.array(images[i])
        if v2:
            rm_bg = RemoveBackgroundV2()
            imgs = rm_bg(img)
            bboxes = []
            for img in imgs:
                bbox = rm_text.detect_text(img)
                if bbox is None:
                    bboxes.append([0, 0, 0, 0])
                else:
                    bboxes.append(transform_bbox(bbox))
            all_bboxes.append(bboxes)

        else: 
            image = np.array(images[i])
            bbox = transform_bbox(rm_text.detect_text(image))
            all_bboxes.append([bbox])

    pickle.dump(all_bboxes, open("text_boxes.pkl", "wb"))


if __name__ == "__main__":
    main()

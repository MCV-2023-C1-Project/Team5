import glob
import os

import numpy as np
from PIL import Image
import pickle

import evaluation_funcs as evalf
from text_detection import TextDetection


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
    path = "data/Week2/qsd1_w2"

    gt_bboxes = pickle.load(open("data/Week2/qsd1_w2/text_boxes.pkl", "rb"))
    images = {
        get_image_name(file): Image.open(file)
        for file in glob.glob(os.path.join(path, "*.jpg"))
    }

    rm_text = TextDetection()
    bboxes = []
    gt_bboxes = [transform_gt_bbox(b[0]) for b in gt_bboxes]
    for i in range(len(images)):
        image = np.array(images[i])
        bbox = transform_bbox(rm_text.detect_text(image))
        bboxes.append(bbox)
    tp, fp, fn, score = evalf.performance_accumulation_window(bboxes, gt_bboxes)
    score /= len(images)
    print('\nMean IoU:', score)
    print('True positives:', tp)
    print('False positives:', fp)
    print('False negatives:', fn)


if __name__ == "__main__":
    main()

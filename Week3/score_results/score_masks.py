import glob
import os

import numpy as np
from PIL import Image
import cv2

import evaluation_funcs as evalf
from bg_removal import RemoveBackgroundV2

def split_name(file: str) -> str:
    return file.replace("\\", "/").split("/")[-1].split(".")[0]

def get_image_name(file: str) -> int:
    return int(split_name(file))

def main():
    path = "data/Week2/qsd2_w2"

    gt_masks = {
        get_image_name(file): cv2.imread(file, cv2.COLOR_BGR2GRAY)
        for file in glob.glob(os.path.join(path, "*.png"))
    }

    images = {
        get_image_name(file): Image.open(file)
        for file in glob.glob(os.path.join(path, "*.jpg"))
    }

    rm_bg = RemoveBackgroundV2()
    masks = {}
    for name, image in images.items():
        image = np.array(image)
        masks.update({name: rm_bg.get_mask(image)})

    pixelTP = 0
    pixelFP = 0
    pixelFN = 0
    pixelTN = 0
    for name, mask in masks.items():
        [
            localPixelTP,
            localPixelFP,
            localPixelFN,
            localPixelTN,
        ] = evalf.performance_accumulation_pixel(mask, gt_masks[name])
        pixelTP = pixelTP + localPixelTP
        pixelFP = pixelFP + localPixelFP
        pixelFN = pixelFN + localPixelFN
        pixelTN = pixelTN + localPixelTN

    [pixelPrecision, _, _, pixelSensitivity] = evalf.performance_evaluation_pixel(
        pixelTP, pixelFP, pixelFN, pixelTN
    )

    pixelF1 = 0
    if (pixelPrecision + pixelSensitivity) != 0:
        pixelF1 = 2 * (
            (pixelPrecision * pixelSensitivity) / (pixelPrecision + pixelSensitivity)
        )

    print(
        f"Precision: {pixelPrecision}, Sensitivity: {pixelSensitivity}, F1: {pixelF1}"
    )


if __name__ == "__main__":
    main()

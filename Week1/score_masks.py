import glob
import os

import cv2

import evaluation_funcs as evalf
from utils import *
from bg_removal import RemoveBackground


def main():
    path = "data/Week1/qsd2_w1"

    gt_masks = {
        get_image_name(file): cv2.imread(file, cv2.COLOR_BGR2GRAY)
        for file in glob.glob(os.path.join(path, "*.png"))
    }

    images = {
        get_image_name(file): cv2.imread(file)
        for file in glob.glob(os.path.join(path, "*.jpg"))
    }

    rm_bg = RemoveBackground()
    masks = {}
    for name, image in images.items():
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
        # _, axs = plt.subplots(1, 2)
        # axs[0].imshow(mask, cmap='gray')
        # axs[0].set_title('Mask')
        # axs[1].imshow(gt_masks[name], cmap='gray')
        # axs[1].set_title('Ground Truth Mask')
        # plt.show()

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

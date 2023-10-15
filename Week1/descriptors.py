import numpy as np
import cv2


class Histogram:
    def __init__(self, color_model="rgb", **kwargs):
        self.color_model = color_model
        self.kwargs = kwargs
        self.color_functions = {
            "rgb": lambda img: img,
            "hsv": lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2HSV),
            "lab": lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2LAB),
            "yuv": lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2YUV),
        }

    def __call__(self, img):
        # change color model if needed
        img = self.color_functions[self.color_model](img)
        # compute histograms over individual channels, then concatenate
        histogram = np.vstack(
            [cv2.calcHist([img], [channel], None, [self.kwargs["bins"]], self.kwargs["range"]) for channel in range(img.shape[2])]
        ).flatten()
        # normalize wrt to the total number of pixels
        histogram = histogram / img.size
        return histogram
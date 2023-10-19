import numpy as np

import cv2


def close(img):
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closing


def erode(img):
    h, w = img.shape
    hper = 0.007
    wper = 0.007
    hker = int(h * hper)
    wker = int(w * wper)

    kernel = np.ones((hker, wker), np.uint8)
    erode = cv2.erode(img, kernel, iterations=5)
    
    return erode


def dilate(img):
    h, w = img.shape
    hper = 0.005
    wper = 0.005
    hker = int(h * hper)
    wker = int(w * wper)

    kernel = np.ones((hker, wker), np.uint8)
    dilate = cv2.dilate(img, kernel, iterations=5)

    return dilate

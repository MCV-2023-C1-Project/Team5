import cv2


class Median:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, image, kernel_size=3):
        return cv2.medianBlur(image, kernel_size)


class Average:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, image, kernel_size=3):
        return cv2.blur(image, [kernel_size, kernel_size])




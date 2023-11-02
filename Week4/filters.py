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


class Bilateral:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, image, d=9, sigmacolor=200, sigmaspace=200):
        return cv2.bilateralFilter(image, d, sigmacolor, sigmaspace)


class Gaussian:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, image, kernel=3, stdX=0):
        return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=stdX)



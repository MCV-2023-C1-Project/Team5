import numpy as np
import cv2
from skimage import feature
import pytesseract


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

    def __call__(self, img, mask=None):
        # change color model if needed
        img = self.color_functions[self.color_model](img)
        # compute histograms over individual channels, then concatenate
        histogram = np.vstack(
            [
                cv2.calcHist(
                    [img], [channel], mask, [self.kwargs["bins"]], self.kwargs["range"]
                )
                for channel in range(img.shape[2])
            ]
        ).flatten()
        # normalize wrt to the total number of pixels
        histogram = histogram / img.size
        return histogram


class SpatialDescriptor:
    def __init__(self, base_descriptor, split_shape):
        """Class for computing a spatial descriptor.

        Args:
            base_descriptor:
                Callable object for computing a descriptor
                for an image (tile).
            split_shape:
                Sequence, first element is a number of vertical
                divisions, second -- horizontal. For example, applying
                split_shape=(1, 3) to a square image results into three
                "tall" subimages.
        """
        self.base_descriptor = base_descriptor
        self.split_shape = split_shape

    def split_in_tiles(self, img):
        """Split an image into tiles.

        Args:
            img: input image, NumPy array of shape (Height, Width,
                Channels).

        Returns:
            A list of subimages of the input image obtained with respect
            to the split_shape.

        Note:
            tiles on bottom/right horizontal/vertical stripes may come
            smaller if the input image can not be equally divided by the
            split shape.
        """

        tiles = np.array_split(img, self.split_shape[0])  # make stripes
        tiles = [
            np.array_split(stripe, self.split_shape[1], 1) for stripe in tiles
        ]  # divide stripes horizontally
        tiles = sum(tiles, [])  # flatten nested list

        return tiles

    def __call__(self, img, mask=None):
        if mask is None:
            tile_descriptors = [
                self.base_descriptor(tile) for tile in self.split_in_tiles(img)
            ]
        else:
            tile_descriptors = [
                self.base_descriptor(img_tile, mask_tile)
                for img_tile, mask_tile in zip(
                    self.split_in_tiles(img), self.split_in_tiles(mask)
                )
            ]

        result = np.concatenate(tile_descriptors)

        return result


class DiscreteCosineTransform:
    def __init__(self, bins: int = 256, range: tuple = (0, 255)) -> None:
        self.bins = bins
        self.range = range

    def __call__(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        histograms = []
        r, c = image.shape[:2]
        r = cv2.getOptimalDFTSize(r)
        c = cv2.getOptimalDFTSize(c)
        dct_image = cv2.resize(image, (c, r))

        for ch in range(image.shape[2]):
            dct = cv2.dct(np.float32(dct_image[:, :, ch]))
            hist = cv2.calcHist([dct], [0], mask, [self.bins], self.range)
            histograms.append(hist)

        histogram = np.vstack(histograms).flatten() / image.size
        return histogram


class LocalBinaryPattern:
    def __init__(
        self,
        numPoints: int,
        radius: int,
        bins: int = 256,
        range: tuple = (0, 255),
        method: str = "uniform",
    ) -> None:
        self.numPoints = numPoints
        self.radius = radius
        self.method = method
        self.bins = bins
        self.range = range

    def __call__(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        histograms = []
        for ch in range(image.shape[2]):
            lbp = feature.local_binary_pattern(
                image[:, :, ch], self.numPoints, self.radius, method=self.method
            ).astype(np.uint8)
            hist = cv2.calcHist([lbp], [0], mask, [self.bins], self.range)
            histograms.append(hist)

        histogram = np.vstack(histograms).flatten() / image.size
        return histogram


class ArtistReader:
    def __init__(self, text_detector, ocr_fn, artists_db=None, distance_fn=None):
        self.text_detector = text_detector
        self.ocr_fn = ocr_fn
        self.artists_db = artists_db
        self.distance_fn = distance_fn

    def __call__(self, img):
        x, y, w, h = self.text_detector.detect_text(img)
        text_img = img[y : y + h, x : x + w]
        text = pytesseract.image_to_string(text_img)
        if self.artists_db is None or self.distance_fn is None:
            return text
        else:
            distances = [self.distance_fn(text, artist) for artist in self.artists_db]
            result = self.artists_db[np.argmin(distances)[0]]
            return result

import numpy as np
import cv2
from skimage import feature
import pytesseract
import re
import pandas as pd
import Levenshtein

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


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
    def __init__(self, base_descriptor, split_shape: tuple):
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
    def __init__(self, bins: int = 8, num_coeff: int = 4) -> None:
        self.bins = bins
        self.num_coeff = num_coeff

    def compute_zig_zag(self, array: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                np.diagonal(array[::-1, :], k)[:: (2 * (k % 2) - 1)]
                for k in range(1 - array.shape[0], array.shape[0])
            ]
        )

    def __call__(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        r, c = image.shape[:2]
        r, c = cv2.getOptimalDFTSize(r), cv2.getOptimalDFTSize(c)
        # Ensure even dimensions
        r += r % 2
        c += c % 2
        grayscale_image = cv2.resize(grayscale_image, (c, r))

        if mask is not None:
            mask = cv2.resize(mask, (c, r))
            grayscale_image = cv2.bitwise_and(
                grayscale_image, grayscale_image, mask=mask
            )

        dct_image = cv2.dct(np.float32(grayscale_image) / 255.0)
        coeffs = self.compute_zig_zag(dct_image[:6, :6])[: self.num_coeff]
        return coeffs


class LocalBinaryPattern:
    def __init__(
        self,
        numPoints: int = 24,
        radius: int = 8,
        method: str = "default",
    ) -> None:
        self.numPoints = numPoints
        self.radius = radius
        self.method = method

    def __call__(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = (
            feature.local_binary_pattern(
                image, self.numPoints, self.radius, method=self.method
            )
        ).astype(np.uint8)
        bins = self.numPoints + 2
        hist = cv2.calcHist([image], [0], mask, [bins], (0, bins))
        hist = cv2.normalize(hist, hist)
        return hist.flatten()


class ArtistReader:
    def __init__(
        self,
        text_detector,
        path_bbdd_csv,
        save_txt_path,
        artists_db=None,
        distance_fn=None,
    ):
        self.text_detector = text_detector
        self.bbdd_path = path_bbdd_csv
        self.txt_path = save_txt_path

        ref_set = pd.read_csv(self.bbdd_path, encoding="ISO-8859-1")
        self.ref_set = {row["idx"]: row["artist"] for _, row in ref_set.iterrows()}

        self.artists_db = artists_db
        self.distance_fn = distance_fn

    def most_similar_string(self, target):
        min_distance = float("inf")
        most_similar = None

        for candidate in self.ref_set.values():
            distance = Levenshtein.distance(target, candidate)
            if distance < min_distance:
                min_distance = distance
                most_similar = candidate

        return most_similar

    def valid_text(self, text):
        clean_text = text.replace(" ", "")
        return len(clean_text) > 0

    def emergency_case(self, image):
        height = image.shape[0]
        text = None
        for i in range(1, 4):
            img_part = image[0 : i * int(height / 4), :]
            text = self.text_detector.read_second_try(img_part)
            if text is None:
                continue
            else:
                break
        return text

    def save_txt(self, text, file_name):
        with open(file_name, "a") as file:
            file.write(f"{text}\n")

    def __call__(self, img):
        try:
            x, y, w, h = self.text_detector.detect_text(img)
        except TypeError:
            return [None]
        text_img = img[y : y + h, x : x + w]
        text = pytesseract.image_to_string(text_img)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        if not self.valid_text(text):
            text = self.emergency_case(img)
        if self.artists_db is None or self.distance_fn is None:
            return self.most_similar_string(text) if self.valid_text(text) else [None]
        else:
            distances = [self.distance_fn(text, artist) for artist in self.artists_db]
            result = self.artists_db[np.argmin(distances)[0]]
            return result


class SIFTExtractor:
    def __init__(self, num_features: int = 0):
        self.sift = cv2.SIFT_create(num_features)

    def __call__(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        keypoints, descriptors = self.sift.detectAndCompute(image, mask=mask)
        return keypoints, descriptors


class ColorSIFTExtractor(SIFTExtractor):
    def __init__(self, num_features: int = 0):
        super().__init__(num_features)

    def __call__(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        b, g, r = cv2.split(image)

        keypoints_b, descriptors_b = self.sift.detectAndCompute(b, mask)
        keypoints_g, descriptors_g = self.sift.detectAndCompute(g, mask)
        keypoints_r, descriptors_r = self.sift.detectAndCompute(r, mask)

        # Concatenate the descriptors for each channel
        keypoints_color = np.concatenate(
            (keypoints_r, keypoints_g, keypoints_b), axis=0
        )
        descriptors_color = np.concatenate(
            (descriptors_r, descriptors_g, descriptors_b), axis=0
        )

        return keypoints_color, descriptors_color


class GLOHExtractor:
    def __init__(self, contrast_threshold: float = 0.03):
        self.gloh = cv2.SIFT_create(contrastThreshold=contrast_threshold)

    def __call__(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        keypoints, descriptors = self.gloh.detectAndCompute(image, mask)
        return keypoints, descriptors


class ORBExtractor:
    def __init__(self):
        self.orb = cv2.ORB_create()

    def __call__(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        keypoints, descriptors = self.orb.detectAndCompute(image, mask)
        return keypoints, descriptors


class KAZEExtractor:
    def __init__(self):
        self.kaze = cv2.KAZE_create()

    def __call__(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        keypoints, descriptors = self.kaze.detectAndCompute(image, mask)
        return keypoints, descriptors

import cv2
import numpy as np


class TextDetection:
    def close_then_open(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        # Put the letters together
        binaryClose1 = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        # Remove noise
        kernel = np.ones((5, 5))
        binary = cv2.morphologyEx(binaryClose1, cv2.MORPH_OPEN, kernel)

        return binary

    def detect_text(self, image: np.ndarray) -> list:
        # Get grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1] < 22

        # Morphological gradients
        kernel = np.ones((4, 4), np.uint8)
        dilate = cv2.dilate(gray, kernel, iterations=1)
        eorde = cv2.erode(gray, kernel, iterations=1)
        gradient = dilate - eorde

        # Gradients of only small saturation
        gradient = gradient * saturation

        # Binarize
        _, binary = cv2.threshold(gradient, 65, 255, cv2.THRESH_BINARY)

        binary1 = self.close_then_open(binary, np.ones((1, int(image.shape[1] / 25))))
        binary2 = self.close_then_open(binary1, np.ones((1, int(image.shape[1] / 5))))
        binary3 = self.close_then_open(binary2, np.ones((1, int(image.shape[1] / 5))))

        return self.get_text_coords(image, binary3)

    def get_text_coords(self, im: np.ndarray, imbw: np.ndarray) -> list:
        bboxes = []
        # Apply Gaussian blur to reduce noise

        kernel = np.ones((4, 4), np.uint8)
        dilate = cv2.dilate(imbw, kernel, iterations=3)
        edges = cv2.Canny(dilate, 30, 70)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            if area > 0.001 * im.size and w > h:
                bboxes.append([x, y, w, h])

        if len(bboxes) == 0:
            return None

        if len(bboxes) == 1:
            return bboxes[0]

        # draw the bounding box into the image
        return self.get_best_bbox(im, bboxes)

    def get_best_bbox(
        self, image: np.ndarray, bboxes: list, threshold: int = 128
    ) -> (int, int, int, int):
        best_bbox = bboxes[0]
        best_score = 0

        for bbox in bboxes:
            x, y, width, height = bbox
            sub_image = image[y : y + height, x : x + width]
            # reader = easyocr.Reader(['en'])
            # text = reader.readtext(sub_image)
            # max_text = 0
            # if len(text) > 0:
            #     best_bbox = bbox
            #     max_text = len(text)
            mean_intensity = np.mean(sub_image)
            score = abs(mean_intensity - threshold)

            if score > best_score:  # and len(text) >= max_text:
                best_score = score
                best_bbox = bbox

        return best_bbox

    def get_text_mask(self, image: np.ndarray) -> np.ndarray:
        bbox_coords = self.detect_text(image)
        if bbox_coords is None:
            return None
        mask = np.full_like(image[:, :, 0], 255, np.uint8)
        x, y, w, h = bbox_coords
        mask[y : y + h, x : x + w] = 0
        return mask

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.get_text_mask(image)


class TextDetectionBypass(TextDetection):
    def get_text_mask(self, image: np.ndarray):
        return None

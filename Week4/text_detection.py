import cv2
import numpy as np
import easyocr
import pytesseract
import re
import easyocr
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


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

    def read_second_try(self, image):
        reader = easyocr.Reader(['en'])
        text = reader.readtext(image)
        return text[0][1] if len(text) > 0 else ''

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.get_text_mask(image)


class TextDetectionBypass(TextDetection):
    def get_text_mask(self, image: np.ndarray):
        return None


class TextDetection_W4:
    def close_then_open(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        # Put the letters together
        binaryClose1 = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        # Remove noise
        kernel = np.ones((5, 5))
        binary = cv2.morphologyEx(binaryClose1, cv2.MORPH_OPEN, kernel)

        return binary

    def open_then_close(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        # Put the letters together
        binaryOpen1 = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        # Remove noise
        kernel = np.ones((5, 5))
        binary = cv2.morphologyEx(binaryOpen1, cv2.MORPH_CLOSE, kernel)

        return binary

    def detect_text(self, image: np.ndarray) -> list:
        # Get grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1] < np.mean(hsv[:, :, 1])

        # Morphological gradients
        kernel = np.ones((4, 4), np.uint8)
        dilate = cv2.dilate(gray, kernel, iterations=1)
        eorde = cv2.erode(gray, kernel, iterations=1)
        gradient = dilate - eorde

        # Gradients of only small saturation
        gradient = gradient * saturation

        # Binarize
        _, binary = cv2.threshold(gradient, np.mean(gray), 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        plt.imshow(binary)
        plt.title("Binary")
        plt.show()
        binary1 = self.close_then_open(binary, np.ones((1, int(image.shape[1] / 25))))

        binary2 = self.close_then_open(binary1, np.ones((1, int(image.shape[1] / 10))))
        plt.imshow(binary2)
        plt.title("Binary 2")
        plt.show()
        # print(f"Binary Mean: {np.mean(binary2)}")

        # binary3 = self.close_then_open(binary2, np.ones((1, int(image.shape[1] / 5))))
        # plt.imshow(binary3)
        # plt.title("Binary 3")
        # plt.show()
        return self.get_text_coords(image, binary2)

    def get_contour_centroid(self, contour):
        e = 0.00000000000000000001
        M = cv2.moments(contour)
        cx = int(M['m10'] / (M['m00'] + e))
        cy = int(M['m01'] / (M['m00'] + e))
        return cx, cy

    def get_text_coords(self, im: np.ndarray, imbw: np.ndarray) -> list:
        bboxes = []
        height, width = im.shape[:2]
        # Apply Gaussian blur to reduce noise

        kernel = np.ones((3, 3), np.uint8)
        dilate = cv2.dilate(imbw, kernel, iterations=3)
        edges = cv2.Canny(dilate, 0, 255)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        parent_contours = [contour for contour, hier in zip(contours, hierarchy[0]) if hier[3] == -1]
        image_contours = im.copy()
        cv2.drawContours(image_contours, parent_contours, -1, (0, 255, 0), 2)

        for contour in parent_contours:
            cx, cy = self.get_contour_centroid(contour)
            if (0.25*width < cx < 0.75*width):
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
            self, image: np.ndarray, bboxes: list, threshold: int = 0
    ) -> (int, int, int, int):
        best_bbox = bboxes[0]
        best_score = 0

        def valid_text(text):
            clean_text = text.replace(" ", "")
            return len(clean_text) > 0

        best_bbox = bboxes[0]
        max_text = 0
        max_contrast = 0
        for bbox in bboxes:
            x, y, width, height = bbox
            sub_image = image[y: y + height, x: x + width]
            text = pytesseract.image_to_string(sub_image)
            if valid_text(text):
                reader = easyocr.Reader(['en'])
                text = reader.readtext(sub_image)
                if len(text) > 0:
                    text = text[0][1]
                    text = re.sub(r'[^a-zA-Z\s]', '', text)
                    if valid_text(text):
                        contrast = np.std(sub_image)
                        normalized_contrast = contrast / sub_image.size
                        max_text = len(text)
                        mean_intensity = np.mean(sub_image)
                        score = abs(mean_intensity - threshold)

                        if normalized_contrast > max_contrast: # and len(text) >= max_text:
                            print(f"Text: {text}")
                            max_contrast = normalized_contrast
                            best_bbox = bbox

        return best_bbox

    def get_text_mask(self, image: np.ndarray) -> np.ndarray:
        bbox_coords = self.detect_text(image)
        if bbox_coords is None:
            return None
        mask = np.full_like(image[:, :, 0], 255, np.uint8)
        x, y, w, h = bbox_coords
        mask[y: y + h, x: x + w] = 0
        plt.imshow(image[y: y + h, x: x + w])
        plt.show()
        return mask

    def read_second_try(self, image):
        reader = easyocr.Reader(['en'])
        text = reader.readtext(image)
        return text[0][1] if len(text) > 0 else ''

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.get_text_mask(image)

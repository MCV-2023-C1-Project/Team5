import numpy as np
import cv2

from utils import *


class Identity:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img


class RemoveBackground:
    def __init__(self, threshold: int = 15, percentage_image: float = 0.3):
        """
        Initializes the RemoveBackground class with the given threshold and percentage_image values.

        Args
            threshold (int): The threshold value used to determine if a pixel is part of the background or not.
            percentage_image (float): The percentage of the image that is considered to be the background.
        """
        self.threshold = threshold
        self.percentage_image = percentage_image

    def get_mask(self, img: np.ndarray) -> np.ndarray:
        """
        Returns a binary mask of the image where the background is white and the foreground is black.

        Args
            img (numpy.ndarray): The input image.

        Returns
            numpy.ndarray: A binary mask of the image where the background is white and the foreground is black.
        """
        height, width = img.shape[:2]
        half_width = int(width / 2)
        half_height = int(height / 2)
        mask = np.zeros((height, width), dtype=np.uint8)

        top_y = int(height * self.percentage_image)
        bottom_y = int(height * (1 - self.percentage_image))
        top_x = int(width * self.percentage_image)
        bottom_x = int(width * (1 - self.percentage_image))

        # Access pixel values at the specified locations
        left = img[half_height, 0, :].astype(np.int32)
        top = img[0, half_width, :].astype(np.int32)
        right = img[half_height, width - 1, :].astype(np.int32)
        bottom = img[height - 1, half_width, :].astype(np.int32)
        start_x, start_y, end_x, end_y = 0, 0, 0, 0

        # Top to bottom
        for y in range(1, top_y):
            dif = np.mean(np.abs(top - img[y, half_width, :]))
            if dif > self.threshold:
                break
            top = img[y, half_width, :].astype(np.int32)
            start_y = y

        # Bottom to top
        for y in range(height - 2, bottom_y, -1):
            dif = np.mean(np.abs(bottom - img[y, half_width, :]))
            if dif > self.threshold:
                break
            bottom = img[y, half_width, :].astype(np.int32)
            end_y = y

        # Left to right
        for x in range(1, top_x):
            dif = np.mean(np.abs(left - img[half_height, x, :]))
            if dif > self.threshold:
                break
            left = img[half_height, x, :].astype(np.int32)
            start_x = x

        # Right to left
        for x in range(width - 2, bottom_x, -1):
            dif = np.mean(np.abs(right - img[half_height, x, :]))
            if dif > self.threshold:
                break
            right = img[half_height, x, :].astype(np.int32)
            end_x = x

        # Create mask
        mask[start_y:end_y, start_x:end_x] = 255

        # full-black masks make errors, so here is the temporary solution
        # TODO: come up with something better
        if mask.sum() == 0:
            mask = np.full_like(mask, 255, np.uint8)

        return mask

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Removes the background from the input image.

        Args
            img (numpy.ndarray): The input image.

        Returns
            numpy.ndarray: The input image with the background removed.
        """
        mask = self.get_mask(img)
        rows, cols = np.where(mask == 255)
        output = img[min(rows) : max(rows) + 1, min(cols) : max(cols) + 1]
        return output


class RemoveBackgroundV2(RemoveBackground):
    # Improved version for multiple images

    def search_middle(self, img: np.ndarray) -> (int, int):
        """
        Finds the middle point of the image by searching for a line of pixels with similar color values.

        Args
            img (np.ndarray): The image to search for the middle point.

        Returns
            (int, int): The x and y coordinates of the middle point.
        """
        height, width = img.shape[:2]
        blocks = 8

        w_points = []
        h_points = []
        for i in range(1, blocks):
            w_points.append(i * width // blocks)
            h_points.append(i * height // blocks)

        top_y = int(height * self.percentage_image)
        top_x = int(width * self.percentage_image)

        w_found = False
        w_mid = 0
        for w in w_points:
            top = img[0, w].astype(np.int32)
            # Top to bottom
            for y in range(1, top_y):
                dif = np.mean(np.abs(top - img[y, w]))
                if dif > self.threshold:
                    break
                top = img[y, w].astype(np.int32)
                if y == top_y - 1:
                    w_found = True
                    w_mid = w
            if w_found:
                break

        h_found = False
        h_mid = 0
        for h in h_points:
            left = img[h, 0].astype(np.int32)
            # Left to right
            for x in range(1, top_x):
                dif = np.mean(np.abs(left - img[h, x]))
                if dif > self.threshold:
                    break
                left = img[h, x].astype(np.int32)
                if y == top_x - 1:
                    h_found = True
                    h_mid = h
            if h_found:
                break

        if w_found:
            return w_mid, 0
        elif h_found:
            return 0, h_mid

    def crop_and_merge_masks(
        self, image: np.ndarray, mid: int, axis: int
    ) -> np.ndarray:
        # Get three channels for the get_mask function
        empty_channel = np.zeros_like(image)
        color_image = cv2.merge((image, empty_channel, empty_channel))

        # Crop the image by the middle to get the pictures on different images
        if axis == 1:
            img1 = color_image[:, :mid]
            img2 = color_image[:, mid:]
        elif axis == 0:
            img1 = color_image[:mid, :]
            img2 = color_image[mid:, :]

        # Get the mask for each image
        mask1 = super().get_mask(img1)
        mask2 = super().get_mask(img2)
        # Concatenate both masks to get the full mask
        mask = np.concatenate((mask1, mask2), axis=axis)

        return mask

    def delete_small_contour(self, image, contour):
        contour_l = []
        for c in contour:
            if cv2.contourArea(c) > 0.05 * image.size:
                contour_l.append(c)

        return contour_l


    def create_mask(self, img: np.ndarray) -> any:
        image = np.array(img)
        edges = cv2.Canny(image, 10, 80)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=5)

        contours, _ = cv2.findContours(
            dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Sort contours by its size
        contours = sorted(
            contours, key=lambda x: cv2.contourArea(x, True), reverse=False
        )[0:2]

        # Get only the two biggest contours
        contours = self.delete_small_contour(image, contours)

        # Get the vertices from the two biggest contours
        v = []
        square = True
        for c in contours:
            temp_v = {}
            peri = cv2.arcLength(c, True)
            vertices = cv2.approxPolyDP(c, peri * 0.04, True)

            if len(vertices) != 4:
                square = False

            x_coords = vertices[:, :, 0]
            y_coords = vertices[:, :, 1]
            temp_v.update(
                {
                    "most_right": np.max(x_coords),
                    "most_left": np.min(x_coords),
                    "most_top": np.min(y_coords),
                    "most_bottom": np.max(y_coords),
                }
            )
            v.append(temp_v)

        """
        CASE 1: Contours obtained but are not rectangles.

        Solution: Find the gap between them and get the mask as done in Week 1

        CASE 2: Contours obtained are rectangles.

        Solution: Fill the mask with that rectangles.
        """
        if not square and len(contours) > 1:
            if v[0]["most_right"] < v[1]["most_left"]:
                mid = int(
                    v[0]["most_right"] + (v[1]["most_left"] - v[0]["most_right"]) / 2
                )
                mask = self.crop_and_merge_masks(image, mid, axis=1)

                return mask

            elif v[1]["most_right"] < v[0]["most_left"]:
                mid = int(
                    v[1]["most_right"] + (v[0]["most_left"] - v[1]["most_right"]) / 2
                )
                mask = self.crop_and_merge_masks(image, mid, axis=1)

                return mask

            elif v[0]["most_bottom"] < v[1]["most_top"]:
                mid = int(
                    v[0]["most_bottom"] + (v[1]["most_top"] - v[0]["most_bottom"]) / 2
                )
                mask = self.crop_and_merge_masks(image, mid, axis=0)

                return mask


        else:
            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

            return mask

        """
            CASE 3: Not good contours enough for positioning the images on the background
            
            Solution: Map the full image trying to find the gap between the paintings and then apply Week 1 techniques
        """
        try:
            w_mid, h_mid = self.search_middle(image)
            if w_mid != 0:
                mask = self.crop_and_merge_masks(image, w_mid, axis=1)
                return mask

            elif h_mid != 0:
                mask = self.crop_and_merge_masks(image, h_mid, axis=0)
                return mask
        except:
            # Get three channels for the get_mask function
            empty_channel = np.zeros_like(image)
            color_image = cv2.merge((image, empty_channel, empty_channel))
            return super().get_mask(color_image)

        print("Not able to detect the images")
        return None


    def separate_image(self, image: np.ndarray, mask: np.ndarray) -> list:
        cropped_images = []
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        gray_masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
        contours, _ = cv2.findContours(gray_masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(
            contours, key=lambda x: cv2.contourArea(x, True), reverse=False
        )[:2]

        def get_x_coordinate(contour):
            x, _, _, _ = cv2.boundingRect(contour)
            return x

        contours = sorted(contours, key=get_x_coordinate, reverse=False)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            if w > image.shape[1]*0.1 and h > image.shape[0]*0.1:
                cropped_images.append(masked_image[y:y+h, x:x+w])

        return cropped_images

    def get_mask(self, img: np.ndarray) -> any:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 1]
        mask = self.create_mask(hsv)

        if mask is None:
            return None

        mask = erode(mask)
        mask = dilate(mask)
        return mask

    def __call__(self, image: np.ndarray) -> list:
        mask = self.get_mask(image)

        if mask is None:
            return np.ones(image.shape[:2], dtype=np.uint8) * 255

        return self.separate_image(image, mask)


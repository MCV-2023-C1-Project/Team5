import numpy as np
import cv2


class Identity:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img


class RemoveBackground:
    def __init__(self, threshold=15, percentage_image=0.3):
        self.threshold = threshold
        self.percentage_image = percentage_image

    def get_mask(self, img):
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
        """Crop images, since masks are rectangular"""
        mask = self.get_mask(img)
        rows, cols = np.where(mask == 255)
        output = img[min(rows) : max(rows) + 1, min(cols) : max(cols) + 1]
        return output

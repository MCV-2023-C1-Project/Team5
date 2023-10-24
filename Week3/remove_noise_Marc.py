import os
from pathlib import Path

import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means

QUERY_IMG_DIR = Path(os.path.join("data", "Week3", "qsd1_w3"))


def bilateral_filter(image):
    return cv2.bilateralFilter(image, 9, 75, 75)  # Adjust parameters as needed

def nl_means(image):
    # Apply Non-Local Means Denoising
    return denoise_nl_means(image, h=0.1, fast_mode=True)

def mean_filter(image):
    return cv2.blur(image, (5, 5))
class ImageNoiseRemoval:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.noise_types = {
            'gaussian': self.detect_gaussian_noise,
            'salt_and_pepper': self.detect_salt_and_pepper_noise,
        }

    def calculate_image_variance(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        variance = np.var(gray)
        return variance

    def detect_histogram_outliers(self, threshold=10):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        hist, _ = np.histogram(gray.flatten(), bins=256, range=[0, 256])
        outlier_percentage = np.sum(hist > threshold) / len(hist)

        if outlier_percentage > 0.02:  # Adjust the threshold as needed
            return True
        return False

    def detect_noise(self):
        variance = self.calculate_image_variance()
        histogram_outliers = self.detect_histogram_outliers()

        # Add more noise detection methods as needed
        # Example: Add methods for specific types of noise detection

        if variance < 100 or histogram_outliers:
            return True
        return False
    def detect_gaussian_noise(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        var = np.var(gray)
        if var < 100:
            return True
        return False

    def detect_salt_and_pepper_noise(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        num_white = np.sum(gray == 255)
        num_black = np.sum(gray == 0)
        total_pixels = gray.size
        if (num_white + num_black) / total_pixels > 0.005:  # Adjust the threshold as needed
            return True
        return False

    def remove_gaussian_noise(self, kernel_size=(5, 5)):
        if self.detect_gaussian_noise():
            print('removing gaussian noise')
            self.image = cv2.GaussianBlur(self.image, kernel_size, 0)

    def remove_salt_and_pepper_noise(self, kernel_size=3):
        if self.detect_noise():
            print('removing salt and pepper noise')
            self.image = cv2.medianBlur(self.image, kernel_size)
    def save_processed_image(self, output_path):
        cv2.imwrite(output_path, self.image)

    def tvd(self):
        # Apply Total Variation Denoising
        if self.detect_noise():
            self.image = denoise_tv_chambolle(self.image, weight=0.2)

    def sharpen_image(self):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        self.image = cv2.filter2D(self.image, -1, kernel)

    def apply_dilation(self):
        kernel = np.ones((5, 5), np.uint8)
        self.image = cv2.dilate(self.image, kernel, iterations=1)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, '..', 'data', 'Week3', 'qsd1_w3')
    input_image_path = os.path.join(data_folder, f'00008.jpg')
    im = cv2.imread(input_image_path)
    noise_removal = ImageNoiseRemoval(input_image_path)
    im1 = noise_removal.image
    #noise_removal.remove_salt_and_pepper_noise()
    #image2 = noise_removal.image.copy()
    #noise_removal.sharpen_image()
    #image3 = noise_removal.image
    #im2 = bilateral_filter(im)
    noise_removal.tvd()
    im3 = noise_removal.image
    noise_removal.sharpen_image()
    im4 = noise_removal.image
    #im4 = nl_means(im)
    combined_image = np.hstack((im, im3, im4))
    #combined_image = np.hstack((im, image2, image3))
    #cv2.imshow('image with noise, without noise and sharpened', combined_image)
    cv2.imshow('image filtered by bilateral filter, tvd', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



import cv2
import numpy as np
import matplotlib.pyplot as plt
from filters import *
from PIL import Image
import os
from pathlib import Path


class Salt_Pepper_Noise:
    def __init__(self, std_mean=9, std_median=7):
        self.std_mean = std_mean
        self.std_median = std_median

    def plot_images(self, image, denoised_image):
        # Display the original and denoised images
        plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray')
        plt.title('Original Image with Noise'), plt.xticks([]), plt.yticks([])

        plt.subplot(1, 2, 2), plt.imshow(denoised_image, cmap='gray')
        plt.title('Denoised Image'), plt.xticks([]), plt.yticks([])

        plt.show()

    def plot_std_dev(self, std_map, idx=0):
        # Flatten the standard deviation values for histogram plotting
        std_values = std_map.flatten()

        # Set a threshold for standard deviation
        threshold = 10.0

        # Plot the histogram
        hist, bins, _ = plt.hist(std_values, bins=25, range=(0, np.max(std_values)), color='blue', alpha=0.7)
        plt.axvline(x=threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold ({threshold})')
        plt.title('Histogram of Standard Deviation Values')
        plt.xlabel('Standard Deviation')
        plt.ylabel('Frequency')

        # Print or use the normalized bin counts as needed
        #plt.legend()
        #plt.show()
        plt.savefig('std/std_{}.png'.format(idx))
        plt.close()

    def pixel_contrast(self, image, row, colum, window=1):
        contrasts = image[row - window:row + window, colum - window:colum + window]
        std_dev = np.std(contrasts)
        return std_dev

    def check_if_contrast(self, image, window=1, idx=0):
        height, width = image.shape[:2]
        contrast_map = np.zeros((height, width), dtype=np.float32)
        for row in range(window, height - window):
            for column in range(window, width - window):
                c = self.pixel_contrast(image, row, column)
                contrast_map[row, column] = c

        """
        print("Mean : {}".format(np.mean(contrast_map)))
        print("Max : {}".format(np.max(contrast_map)))
        print("Median : {}".format(np.median(contrast_map)))
        """

        #self.plot_images(image, contrast_map)
        #self.plot_std_dev(contrast_map, idx=idx)

        return True if np.median(contrast_map) >= 7 and np.mean(contrast_map) >= 9 else False

    def __call__(self, image, window=1, idx=0):
        return self.check_if_contrast(image, window=window, idx=idx)



if __name__ == "__main__":

    QUERY_IMG_DIR = Path(os.path.join("data", "Week3", "qsd1_w3"))

    NOISE_FILTER = Median()
    HAS_NOISE = Salt_Pepper_Noise()

    for img_path in QUERY_IMG_DIR.glob("*.jpg"):
        idx = int(img_path.stem[-5:])
        print("Image {}".format(idx))
        img = Image.open(img_path)
        image = np.array(img)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 2]
        # Load an image
        #image = cv2.imread('data/Week3/qsd1_w3/00002.jpg')

        # Check if the image has salt-and-pepper noise
        has_salt_and_pepper = HAS_NOISE(hsv, idx=idx)

        if has_salt_and_pepper:
            kernel_size_factor = 0.008
            # Adjust kernel size based on image dimensions
            kernel_size = tuple(int(dim * kernel_size_factor) for dim in image.shape)
            # Ensure that the kernel size is odd
            kernel_size = tuple(size + 1 if size % 2 == 0 else size for size in kernel_size)

            denoised_image = NOISE_FILTER(image, kernel_size=3)

            #plot_images(image, denoised_image)
            Image.fromarray(denoised_image).save("denoised/{}.png".format(idx))
            print("Denoised Image")
        else:
            Image.fromarray(image).save("denoised/{}.png".format(idx))
            print("Original Image")




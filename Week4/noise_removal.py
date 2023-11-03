from filters import *
import os
from text_detection import *
from PIL import Image
from similar_artist import *
import matplotlib.pyplot as plt
from skimage.restoration import estimate_sigma


class SaltPepperNoise:
    def __init__(self, std_mean=9, std_median=7,
                 noise_filter=None, name_filter=None,
                 text_detector=None):
        self.std_mean = std_mean
        self.std_median = std_median
        self.noise_filter = noise_filter
        self.text_detector = text_detector
        self.name_filter = name_filter

    def plot_images(self, image, denoised_image, final_image):
        # Display the original and denoised images
        plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray')
        plt.title('Original Image with Noise'), plt.xticks([]), plt.yticks([])

        plt.subplot(1, 3, 2), plt.imshow(denoised_image, cmap='gray')
        plt.title('Only Median Filter'), plt.xticks([]), plt.yticks([])

        plt.subplot(1, 3, 2), plt.imshow(final_image, cmap='gray')
        plt.title('Medina filter + Average for name'), plt.xticks([]), plt.yticks([])

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
        plt.savefig('std/std_{}.png'.format(idx))
        plt.close()

    def pixel_contrast(self, image, row, colum, window=1):
        contrasts = image[row - window:row + window, colum - window:colum + window]
        std_dev = np.std(contrasts)
        return std_dev

    def check_if_contrast(self, image, window=1, idx=0):
        height, width = image.shape[:2]
        contrast_map = np.zeros((height, width), dtype=np.float32)
        average_noise = estimate_sigma(image, channel_axis=-1, average_sigmas=True)
        # for row in range(window, height - window):
        #     for column in range(window, width - window):
        #         c = self.pixel_contrast(image, row, column)
        #         contrast_map[row, column] = c
        # #self.plot_images(image, contrast_map)
        # #self.plot_std_dev(contrast_map, idx=idx)

        return True if average_noise >= 9 else False


    def kernel_size(self, image, size_factor=0.008):
        kernel_size = tuple(int(dim * size_factor) for dim in image.shape)
        # Minimum kernel size = 3x3
        kernel_size = tuple(size + 2 if size < 2 else size for size in kernel_size)
        # Get only odd kernels
        return tuple(size + 1 if size % 2 == 0 else size for size in kernel_size)

    def sharp_image(self, image):
        # Apply Laplacian kernel for image sharpening
        laplacian_kernel = np.array([[0, -1, 0],
                                     [-1, 5, -1],
                                     [0, -1, 0]])

        return cv2.filter2D(image, -1, laplacian_kernel)

    def filter_name(self, original_image, image):
        bbox_coords = self.text_detector.detect_text(image)
        if bbox_coords is None:
            return image
        x, y, w, h = bbox_coords
        name_img = original_image[y: y + h, x: x + w]
        den_name = self.name_filter(name_img, kernel_size=3)
        den_name = self.sharp_image(den_name)
        image[y: y + h, x: x + w] = den_name
        # self.plot_images(original_image, image1, image)
        return image


    def __call__(self, image, window=1, idx=0):
        original_image = image
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 2]
        # kernel = self.kernel_size(image)
        if self.check_if_contrast(hsv, window=window):
            image = self.filter_name(original_image, self.noise_filter(image, kernel_size=3))

        return image


# if __name__ == "__main__":

#     QUERY_IMG_DIR = Path(os.path.join("data", "Week3", "testing"))

#     NOISE_FILTER = Median()
#     HAS_NOISE = SaltPepperNoise()
#     Text_Detection = TextDetection()

#     for img_path in QUERY_IMG_DIR.glob("*.jpg"):
#         idx = int(img_path.stem[-5:])
#         img = Image.open(img_path)
#         image = np.array(img)

#         # Check if the image has salt-and-pepper noise
#         denoised_image = HAS_NOISE(image)

#         if True:
#             Image.fromarray(denoised_image).save("denoised/qsd1/{}.png".format(idx))
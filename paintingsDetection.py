import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
def detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow('blurred', blurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    cv2.imshow('edges', edges)
    # Find contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask to draw lines on
    mask = np.zeros_like(edges)

    # Find horizontal and vertical regions without edges
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Check if the contour is horizontal (more width than height)
        if w > h:
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Check if the contour is vertical (more height than width)
        if h > w:
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Draw horizontal and vertical lines on the image
    horizontal_lines = np.max(mask, axis=1)
    vertical_lines = np.max(mask, axis=0)

    # Draw the lines on the image
    for i in range(horizontal_lines.shape[0]):
        if horizontal_lines[i] == 255:
            cv2.line(image, (0, i), (image.shape[1], i), (0, 0, 255), 2)

    for i in range(vertical_lines.shape[0]):
        if vertical_lines[i] == 255:
            cv2.line(image, (i, 0), (i, image.shape[0]), (0, 0, 255), 2)

    # Crop the image based on line coordinates
    x1, y1 = 0, 0  # Top-left corner
    x2, y2 = image.shape[1], image.shape[0]  # Bottom-right corner

    # Adjust the cropping coordinates based on your requirements
    # For example, you can use the coordinates where the lines intersect

    cropped_image = image[y1:y2, x1:x2]

    # Save or display the cropped image
    cv2.imshow('Cropped Image', cropped_image)
    cv2.imwrite('cropped_image.jpg', cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_detection():
    image = cv2.imread('data/Week2/qsd2_w2/00014.jpg')
    #print image in screen for two seconds
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    detected = detection(image)
    cv2.imshow('image', detected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

test_detection()
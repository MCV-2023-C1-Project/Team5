import cv2
import numpy as np
from pathlib import Path
from PIL import Image

original_path = Path(r"data\Week3\qsd2_w3\\non_augmented")
denoised_path = Path(r"denoised\qsd2")



original_set = {}
for img_path in original_path.glob("*.jpg"):
    idx = int(img_path.stem[-5:])
    img = Image.open(img_path)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    original_set[idx] = img  # add "idx: descriptor" pair

denoised_set = {}
for img_path in denoised_path.glob("*.png"):
    idx = int(img_path.stem)
    img = Image.open(img_path)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised_set[idx] = img  # add "idx: descriptor" pair

print(denoised_set.keys())

f1_scores = []
for idx in denoised_set.keys():
    print(idx)
    # Apply thresholding using Otsu's method
    _, original_mask = cv2.threshold(original_set[idx], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, denoised_mask = cv2.threshold(denoised_set[idx], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Compute TP, FP, FN
    TP = np.sum(np.logical_and(original_mask, denoised_mask))
    FP = np.sum(np.logical_and(np.logical_not(original_mask), denoised_mask))
    FN = np.sum(np.logical_and(original_mask, np.logical_not(denoised_mask)))

    # Compute precision, recall, and F1 score
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    f1_scores.append(f1_score)


# Compute average F1 score
average_f1_score = np.mean(f1_scores)

# Print or store the results
print(f'Average F1 Score: {average_f1_score:.4f}')





import Levenshtein
import pandas as pd
from pathlib import Path
import pytesseract
import re
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class SimilarArtist:
    def __init__(self, path_query_csv):
        self.query_path = path_query_csv

    def most_similar_string(self, target, ref_set):
        min_distance = float('inf')
        most_similar = None

        ref_set = {row["idx"]: row["artist"] for _, row in ref_set.iterrows()}
        for candidate in ref_set.values():
            distance = Levenshtein.distance(target, candidate)
            if distance < min_distance:
                min_distance = distance
                most_similar = candidate

        #print(f"The most similar string to '{target}' is: '{most_similar}'")
        return most_similar


    def text_similarities(self, text):
        ref_set = pd.read_csv(self.query_path)

        return self.most_similar_string(text, ref_set)

    def valid_text(self, text):
        clean_text = text.replace(" ", "")
        return len(clean_text) > 0

    def __call__(self, image, bbox_coords):
        x, y, w, h = bbox_coords
        image = image[y: y + h, x: x + w]
        text = pytesseract.image_to_string(image)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return self.text_similarities(text) if self.valid_text(text) else [None]


# if __name__ == "__main__":
#     target_string = "Ponce"
#     REF_CSV_PATH = Path(r"C:\Users\maria\PycharmProjects\C1_CVC\data\Week3\paintings_db_w2d1.csv")
#     ref_set = pd.read_csv(REF_CSV_PATH)
#     Similar_Artist = SimilarArtist(REF_CSV_PATH)

#     most_similar = Similar_Artist.most_similar_string(target_string, ref_set)


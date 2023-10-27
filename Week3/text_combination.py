import pandas as pd
from pathlib import Path


class CompareArtist:
    def __init__(self, path_csv_bbdd):
        self.bbdd_path = path_csv_bbdd

        ref_set = pd.read_csv(self.bbdd_path, encoding='ISO-8859-1')
        self.ref_set = {row["idx"]: row["artist"] for _, row in ref_set.iterrows()}

    def get_artist_bbdd(self, artist):
        return [key for key, value in self.ref_set.items() if value == artist]

    def compare_artist(self, results: dict, artists: dict):
        compared_result = []
        for idx, result in results.items():
            if artists[idx] is [None]:
                compared_result.append(result)
                continue
            ref_idx = self.get_artist_bbdd(artists[idx])

            correct_artist = []
            not_correct_artist = []
            for item in result:
                if item in ref_idx:
                    correct_artist.append(item)
                else:
                    not_correct_artist.append(item)
            compared_result.append(correct_artist + not_correct_artist)

        return compared_result

    def compare_multiple_paintings_artist(self, results: dict, artists: dict):
        compared_results = []
        for idx, result in results.items():
            compared_result = []
            for i, artist in enumerate(artists[idx]):
                if artist is [None]:
                    compared_result.append(result[i])
                    continue
                ref_idx = self.get_artist_bbdd(artist)
                correct_artist = []
                not_correct_artist = []
                for item in result[i]:
                    if item in ref_idx:
                        correct_artist.append(item)
                    else:
                        not_correct_artist.append(item)
                compared_result.append(correct_artist + not_correct_artist)
            compared_results.append(compared_result)

        return compared_results




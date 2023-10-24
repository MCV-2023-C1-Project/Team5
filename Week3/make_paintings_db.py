
from pathlib import Path
import re
import csv


DB_DIR = Path(r"C:\Users\krupa\Desktop\qsd1_w2\qsd1_w2")
OUT_CSV_PATH = Path(r"D:\Team5\Week3\paintings_db_w2d1.csv")

def get_artist_and_work(text):
    pattern = r"['\"]([^'\"]+)['\"], ['\"]([^'\"]+)['\"]"
    matches = re.findall(pattern, text)
    if matches:
        artist = matches[0][0]
        work = matches[0][1]
        return artist, work
    else:
        return None, None

with OUT_CSV_PATH.open(mode="w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["idx", "artist", "painting"])
    for txt_path in DB_DIR.glob("*.txt"):
        text = txt_path.read_text(errors="replace")
        idx = int(txt_path.stem[-5:])
        csv_writer.writerow([idx, *get_artist_and_work(text)])



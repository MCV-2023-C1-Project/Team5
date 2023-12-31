import pandas as pd
from pathlib import Path
import numpy as np
import os
from utils import *


# set paths
PRED_RET_PATH = Path("results_10_sift_th0075.pkl")
GT_RET_PATH = Path(os.path.join("data", "qsd1_w4", "gt_corresps.pkl"))

pred = pd.read_pickle(PRED_RET_PATH)

from itertools import chain

pred_chain = [list(chain(*inner_list)) for inner_list in pred]

gt = pd.read_pickle(GT_RET_PATH)

print("Results: ", pred)
print("GT: ", gt)
print("MAPK: ", mapk(gt, pred_chain, k=10))
print("MULTI MAPK: ", multi_mapk(gt, pred, k=10))
print("F1 SCORE: ", f1_score(pred, gt))

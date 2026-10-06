import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import SPLITS, N_FOLDS, N_REPEATS, RANDOM_STATE


def make_fold_table(dev):
    """One row per development segment: its class and its fold number (0..9) in each repeat.

    Folds are made on segments, not chunks, so all 23 chunks of a segment
    always sit in the same fold. Each repeat uses a different shuffle.
    """
    table = dev.groupby("segment")["y"].first().reset_index()
    for r in range(N_REPEATS):
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE + r)
        fold = np.empty(len(table), dtype=int)
        for k, (_, val_idx) in enumerate(skf.split(table["segment"], table["y"])):
            fold[val_idx] = k
        table[f"repeat_{r}"] = fold
    return table


def load_fold_table():
    """The saved fold table. Always load this instead of re-making it."""
    return pd.read_csv(SPLITS / "cv_folds.csv")


def fold_of_rows(dev, table, repeat):
    """Fold number of every chunk row, taken from its segment's fold."""
    return dev["segment"].map(table.set_index("segment")[f"repeat_{repeat}"]).to_numpy()
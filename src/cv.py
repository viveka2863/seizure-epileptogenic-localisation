import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

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


def load_fold_table(name="cv_folds.csv"):
    """The saved fold table. Always load this instead of re-making it."""
    return pd.read_csv(SPLITS / name)


def fold_of_rows(dev, table, repeat):
    """Fold number of every chunk row, taken from its segment's fold."""
    return dev["segment"].map(table.set_index("segment")[f"repeat_{repeat}"]).to_numpy()


def make_group_fold_table(dev, group_of, seed=RANDOM_STATE):
    """Like make_fold_table, but whole groups of twin segments always sit in the same fold.

    seed: base shuffle seed (repeat r uses seed + r). The saved tables use the default.
    """
    table = dev.groupby("segment")["y"].first().reset_index()
    table["group"] = table["segment"].map(group_of)
    for r in range(N_REPEATS):
        skf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed + r)
        fold = np.empty(len(table), dtype=int)
        for k, (_, val_idx) in enumerate(skf.split(table["segment"], table["y"], groups=table["group"])):
            fold[val_idx] = k
        table[f"repeat_{r}"] = fold
    return table
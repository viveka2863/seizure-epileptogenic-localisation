import numpy as np
import pandas as pd

from src.config import RAW, TARGET_CLASSES, RANDOM_STATE, HOLDOUT_FRAC


def load_raw():
    """Load the Kaggle CSV and split the ID column into chunk and segment."""
    df = pd.read_csv(RAW).copy()
    id_col = df.columns[0]                       # unnamed ID column, e.g. X21.V1.791
    parts = df[id_col].str.extract(r"^X(\d+)\.(.+)$")
    df = df.rename(columns={id_col: "row_id"})
    df["chunk"] = parts[0].astype(int)           # 1..23
    df["segment"] = parts[1]                     # source segment, e.g. V1.791
    return df


def feature_cols(df):
    """Names of the 178 signal columns X1..X178."""
    return [c for c in df.columns if c.startswith("X") and c[1:].isdigit()]


def load_binary(classes=TARGET_CLASSES):
    """Rows for the chosen classes only."""
    df = load_raw()
    return df[df["y"].isin(classes)].reset_index(drop=True)


def split_holdout(df, frac=HOLDOUT_FRAC, seed=RANDOM_STATE):
    """Hold out whole segments (never single chunks), the same share per class."""
    rng = np.random.default_rng(seed)
    held = []
    for _, sub in df.groupby("y"):
        segs = sorted(sub["segment"].unique())
        n = int(round(frac * len(segs)))
        held += list(rng.choice(segs, size=n, replace=False))
    mask = df["segment"].isin(held)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)
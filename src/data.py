import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import RAW, PROCESSED, SPLITS, TARGET_CLASSES, RANDOM_STATE, HOLDOUT_FRAC

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


def split_holdout_by_group(df, group_of, frac=HOLDOUT_FRAC, seed=RANDOM_STATE):
    """Hold out whole groups of twin segments: about frac of the segments, roughly balanced by class."""
    seg = df.groupby("segment")["y"].first().reset_index()
    seg["group"] = seg["segment"].map(group_of)
    skf = StratifiedGroupKFold(int(round(1 / frac)), shuffle=True, random_state=seed)
    _, held_idx = next(iter(skf.split(seg["segment"], seg["y"], groups=seg["group"])))
    held = set(seg["segment"].iloc[held_idx])
    mask = df["segment"].isin(held)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def write_processed():
    """Rebuild data/processed/dev_2v3.csv and holdout_2v3.csv from the raw CSV and the saved hold-out list."""
    both = load_binary()
    hold_ids = set(pd.read_csv(SPLITS / "holdout_segments.csv")["segment"])
    mask = both["segment"].isin(hold_ids)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    both[~mask].reset_index(drop=True).to_csv(PROCESSED / "dev_2v3.csv", index=False)
    both[mask].reset_index(drop=True).to_csv(PROCESSED / "holdout_2v3.csv", index=False)


def write_processed_allclass():
    """Rebuild data/processed/dev_allclass.csv and holdout_allclass.csv (all five classes) from the raw CSV and the saved hold-out list."""
    both = load_raw()
    hold_ids = set(pd.read_csv(SPLITS / "holdout_segments_allclass.csv")["segment"])
    mask = both["segment"].isin(hold_ids)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    both[~mask].reset_index(drop=True).to_csv(PROCESSED / "dev_allclass.csv", index=False)
    both[mask].reset_index(drop=True).to_csv(PROCESSED / "holdout_allclass.csv", index=False)
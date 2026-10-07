import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix,
                             precision_score, f1_score)

from src.config import N_FOLDS, N_REPEATS
from src.cv import fold_of_rows


def metrics_from_segments(y_true, p, fold, threshold=0.5):
    """All the scores, from one probability per segment (y_true: True = positive class)."""
    y_true, p, fold = np.asarray(y_true), np.asarray(p), np.asarray(fold)
    y_pred = p > threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    seg = pd.DataFrame({"target": y_true, "p": p, "fold": fold})
    folds = [g for _, g in seg.groupby("fold") if g["target"].nunique() == 2]     # folds that contain both classes
    return {
        "auc": roc_auc_score(y_true, p),
        "auc_fold": float(np.mean([roc_auc_score(g["target"], g["p"]) for g in folds])),
        "pr_auc": average_precision_score(y_true, p),
        "pr_auc_fold": float(np.mean([average_precision_score(g["target"], g["p"]) for g in folds])),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "accuracy": (tp + tn) / len(seg),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }


# ---- chunk-level version: rows are 1-second chunks, predictions are averaged per segment ----

def out_of_fold_proba(make_model, X, target, rows_fold):
    """Probability of the positive class for every chunk, from a model that never saw that chunk's segment."""
    proba = np.empty(len(target))
    for k in range(N_FOLDS):
        train, val = rows_fold != k, rows_fold == k
        model = make_model().fit(X[train], target[train])
        col = list(model.classes_).index(True)
        proba[val] = model.predict_proba(X[val])[:, col]
    return proba


def score(dev, target, proba, rows_fold, threshold=0.5):
    """Segment-level scores: average each segment's chunk probabilities, then score the segments."""
    seg = (pd.DataFrame({"segment": dev["segment"].to_numpy(), "target": target, "fold": rows_fold, "p": proba})
           .groupby("segment").agg(target=("target", "first"), fold=("fold", "first"), p=("p", "mean")))
    out = metrics_from_segments(seg["target"], seg["p"], seg["fold"], threshold)
    out["chunk_accuracy"] = ((proba > threshold) == target).mean()
    return out


def cross_validate(make_model, dev, X, table, n_repeats=N_REPEATS, positive=2, threshold=0.5):
    """Repeated grouped CV on chunk rows, one row of scores per repeat.

    positive: the class that counts as the positive (1 = seizure; 2 for class 2 vs 3).
    Everything else in dev counts as negative. Chunk probabilities are averaged per segment.
    """
    target = (dev["y"] == positive).to_numpy()
    rows = []
    for r in range(n_repeats):
        rows_fold = fold_of_rows(dev, table, r)
        proba = out_of_fold_proba(make_model, X, target, rows_fold)
        rows.append({"repeat": r, **score(dev, target, proba, rows_fold, threshold)})
    return pd.DataFrame(rows)


# ---- segment-level version: rows are segments (one row of features each) ----

def cross_validate_segments(make_model, data, feature_cols, table, n_repeats=N_REPEATS, positive=1, threshold=0.5):
    """Repeated grouped CV on a table with one row per segment, using the same saved folds."""
    X = data[feature_cols].to_numpy(dtype=float)
    target = (data["y"] == positive).to_numpy()
    fold_lookup = table.set_index("segment")
    rows = []
    for r in range(n_repeats):
        fold = data["segment"].map(fold_lookup[f"repeat_{r}"]).to_numpy()
        p = np.full(len(data), np.nan)
        for k in range(N_FOLDS):
            train, val = fold != k, fold == k
            if not val.any():
                continue
            model = make_model().fit(X[train], target[train])
            p[val] = model.predict_proba(X[val])[:, list(model.classes_).index(True)]
        rows.append({"repeat": r, **metrics_from_segments(target, p, fold, threshold)})
    return pd.DataFrame(rows)
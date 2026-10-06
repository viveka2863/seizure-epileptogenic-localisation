import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

from src.config import N_FOLDS, N_REPEATS
from src.cv import fold_of_rows

POSITIVE = 2   # class 2 (the affected area) is the "positive" class


def out_of_fold_proba(make_model, X, y, rows_fold):
    """Probability of class 2 for every chunk, from a model that never saw that chunk's segment."""
    proba = np.empty(len(y))
    for k in range(N_FOLDS):
        train, val = rows_fold != k, rows_fold == k
        model = make_model().fit(X[train], y[train])
        col = list(model.classes_).index(POSITIVE)
        proba[val] = model.predict_proba(X[val])[:, col]
    return proba


def score(dev, proba):
    """Segment-level scores: average each segment's chunk probabilities, then score the segments."""
    seg = (pd.DataFrame({"segment": dev["segment"].to_numpy(), "y": dev["y"].to_numpy(), "p": proba})
           .groupby("segment").agg(y=("y", "first"), p=("p", "mean")))
    is_pos = (seg["y"] == POSITIVE).to_numpy()
    pred_pos = (seg["p"] > 0.5).to_numpy()
    tn, fp, fn, tp = confusion_matrix(is_pos, pred_pos, labels=[False, True]).ravel()
    chunk_acc = ((proba > 0.5) == (dev["y"].to_numpy() == POSITIVE)).mean()
    return {"auc": roc_auc_score(is_pos, seg["p"]), "accuracy": (tp + tn) / len(seg),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn, "chunk_accuracy": chunk_acc}


def cross_validate(make_model, dev, X, table, n_repeats=N_REPEATS):
    """Repeated 10-fold CV on segments. One row of scores per repeat."""
    y = dev["y"].to_numpy()
    rows = []
    for r in range(n_repeats):
        proba = out_of_fold_proba(make_model, X, y, fold_of_rows(dev, table, r))
        rows.append({"repeat": r, **score(dev, proba)})
    return pd.DataFrame(rows)
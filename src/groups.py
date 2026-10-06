import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import squareform

from src.config import SIMILARITY_STRONG, SIMILARITY_WEAK


def similarity_matrix(df, cols):
    """How alike every pair of segments is: mean correlation of chunk k with chunk k, over the 23 chunks."""
    df = df.sort_values(["segment", "chunk"])
    names = sorted(df["segment"].unique())
    A = df[cols].to_numpy(dtype=float).reshape(len(names), 23, len(cols))   # segments x chunks x samples
    A = A - A.mean(axis=2, keepdims=True)
    Z = A / np.linalg.norm(A, axis=2, keepdims=True)
    M = np.einsum("akd,bkd->ab", Z, Z) / 23
    np.fill_diagonal(M, 0)
    return names, M


def make_groups(names, M, strong=SIMILARITY_STRONG, weak=SIMILARITY_WEAK):
    """Group segments that are twins.

    Two segments are linked if they are more similar than `strong`, or if they sit in the same tight
    cluster (complete linkage: every pair inside it above `weak`). Linked segments form one group.
    """
    D = 1 - M
    np.fill_diagonal(D, 0)
    tree = linkage(squareform(D, checks=False), method="complete")
    cluster = fcluster(tree, t=1 - weak, criterion="distance")
    linked = (M > strong) | (cluster[:, None] == cluster[None, :])
    _, labels = connected_components(linked, directed=False)
    return pd.Series(labels, index=names, name="group")
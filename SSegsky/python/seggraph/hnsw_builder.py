from __future__ import annotations

import numpy as np


def build_hnsw_faiss(X: np.ndarray, M: int, efC: int, metric: str):
    """
    Build a Faiss HNSW index on a segment.

    Args:
        X: [n, d] float32 matrix
        M: HNSW connectivity parameter
        efC: efConstruction
        metric: 'l2' or 'ip'
    """
    import faiss

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}")
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")
    if efC <= 0:
        raise ValueError(f"efC must be positive, got {efC}")

    d = X.shape[1]

    if metric == "l2":
        index = faiss.IndexHNSWFlat(d, int(M))
    elif metric == "ip":
        index = faiss.IndexHNSWFlat(d, int(M), faiss.METRIC_INNER_PRODUCT)
    else:
        raise ValueError("metric must be 'l2' or 'ip'")

    index.hnsw.efConstruction = int(efC)
    index.add(X)
    return index
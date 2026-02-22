from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def topk_cosine(query_vec, ref_matrix, ref_ids, k: int = 5):
    """
    Compute cosine similarity between query_vec (1 x d) and ref_matrix (n x d).
    Returns top-k (ids, scores) sorted descending.
    """
    if k <= 0:
        return [], []

    # cosine_similarity returns (1, n)
    sims = cosine_similarity(query_vec, ref_matrix).ravel()

    if len(sims) == 0:
        return [], []

    k = min(k, len(sims))
    top_idx = np.argpartition(-sims, kth=k-1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    ids = [ref_ids[i] for i in top_idx.tolist()]
    scores = [float(sims[i]) for i in top_idx.tolist()]
    return ids, scores
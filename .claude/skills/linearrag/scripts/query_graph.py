# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy>=2.0",
#     "scipy>=1.14",
#     "spacy>=3.8,<3.9",
#     "sentence-transformers>=3.0",
#     "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#     "ja-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/ja_core_news_sm-3.8.0/ja_core_news_sm-3.8.0-py3-none-any.whl",
# ]
# ///
# NOTE: standalone script; wider Python range than the project on purpose
"""LinearRAG two-stage retrieval (paper §3.2): entity activation via semantic
bridging, then passage retrieval via personalized PageRank.

Implemented from the paper's equations (arXiv:2510.10114), independently of
the reference implementation.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def initial_activation(query_entity_vecs: np.ndarray,
                       emb_entities: np.ndarray) -> np.ndarray:
    """Paper eq.3: each query entity activates its most similar graph entity."""
    a0 = np.zeros(emb_entities.shape[0], dtype=np.float32)
    if query_entity_vecs.size == 0 or emb_entities.size == 0:
        return a0
    sims = emb_entities @ query_entity_vecs.T          # |E| x |Eq|
    for j in range(sims.shape[1]):
        i = int(np.argmax(sims[:, j]))
        a0[i] = max(a0[i], float(sims[i, j]))
    return a0


def activate_entities(a0: np.ndarray, M: sparse.csr_matrix, sigma: np.ndarray,
                      delta: float = 0.5, max_iterations: int = 4):
    """Paper eq.5 with dynamic pruning.

    a_t = MAX(M^T (sigma ⊙ (M a_{t-1})), a_{t-1})

    Newly activated entities survive only if their score reaches delta; the
    loop stops as soon as an iteration activates nothing new.

    Returns (scores, levels, trace): levels[i] is the 1-based iteration at
    which entity i first activated (0 = inactive; seeds are level 1) — used
    as L_ei in eq.7. trace lists newly activated entity indices per iteration.
    """
    a = a0.astype(np.float32).copy()
    levels = np.where(a > 0, 1, 0).astype(np.int32)
    sigma = np.clip(sigma, 0.0, None).astype(np.float32)
    trace: list[list[int]] = []

    for it in range(2, max_iterations + 2):
        spread = M.T @ (sigma * (M @ a))
        updated = np.maximum(a, spread)
        newly = (levels == 0) & (updated >= delta)
        a = np.where(levels > 0, updated, np.where(newly, updated, 0.0)).astype(np.float32)
        if not newly.any():
            break
        levels[newly] = it
        trace.append(np.flatnonzero(newly).tolist())
    return a, levels, trace

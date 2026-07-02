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
    """Paper eq.3: each query entity activates its most similar graph entity.

    Both matrices must be L2-normalized (cosine similarity = dot product).
    """
    a0 = np.zeros(emb_entities.shape[0], dtype=np.float32)
    if query_entity_vecs.size == 0 or emb_entities.size == 0:
        return a0
    sims = emb_entities @ query_entity_vecs.T          # |E| x |Eq|
    for j in range(sims.shape[1]):
        i = int(np.argmax(sims[:, j]))
        a0[i] = max(a0[i], float(sims[i, j]))
    np.clip(a0, 0.0, None, out=a0)
    return a0


def activate_entities(
        a0: np.ndarray, M: sparse.csr_matrix, sigma: np.ndarray,
        delta: float = 0.5, max_iterations: int = 4,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Paper eq.5 with dynamic pruning.

    a_t = MAX(M^T (sigma ⊙ (M a_{t-1})), a_{t-1})

    Newly activated entities survive only if their score reaches delta; the
    loop stops as soon as an iteration activates nothing new.

    Returns (scores, levels, trace): levels[i] is the 1-based iteration at
    which entity i first activated (0 = inactive; seeds are level 1) — used
    as L_ei in eq.7. trace lists newly activated entity indices per iteration.
    Score magnitudes are unbounded; downstream consumers absorb this via
    log1p / sum normalization.
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


def _min_max(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:  # degenerate range: all passages equally ranked by DPR
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def passage_seed_scores(sim_qp: np.ndarray, C: sparse.csr_matrix,
                        activation: np.ndarray, levels: np.ndarray,
                        lam: float = 1.5, w_p: float = 0.05) -> np.ndarray:
    """Paper eq.7: hybrid initialization of passage nodes.

    score(p) = (lam * minmax(sim(q,p)) + ln(1 + sum_i a_i * ln(1+N_pi) / L_i)) * w_p
    where N_pi is the occurrence count of entity i in passage p and L_i is the
    activation level from stage 1 (Task 8's levels; 0 = inactive).
    """
    dpr = _min_max(sim_qp.astype(np.float64))
    C_log = C.astype(np.float64)
    C_log.data = np.log1p(C_log.data)
    inv_level = np.divide(activation.astype(np.float64),
                          np.maximum(levels, 1),
                          out=np.zeros_like(activation, dtype=np.float64),
                          where=levels > 0)
    bonus = np.asarray(C_log @ inv_level).ravel()
    return ((lam * dpr + np.log1p(bonus)) * w_p).astype(np.float32)


def personalized_pagerank(B: sparse.csr_matrix, passage_seeds: np.ndarray,
                          entity_seeds: np.ndarray, damping: float = 0.5,
                          max_iter: int = 100, tol: float = 1e-9):
    """Paper eq.6: PPR over the passage-entity bipartite graph via power
    iteration. B is the binarized |P| x |E| biadjacency matrix. The reset
    distribution is the normalized concatenation of the two seed vectors.

    Isolated nodes (deg=0) are set to deg=1 with an empty transition column
    (sink); no dangling correction is applied — the (1-d) reset term keeps the
    distribution well-behaved, though x.sum() may fall marginally below 1 when
    B contains isolated nodes. damping default 0.5 per the reference config.

    Returns (passage_scores, entity_scores).
    """
    n_p, n_e = B.shape
    n = n_p + n_e
    A = sparse.bmat([[None, B], [B.T, None]], format="csr")
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    W = (sparse.diags(1.0 / deg) @ A).T.tocsr()   # column-stochastic transition

    seeds = np.concatenate([passage_seeds, entity_seeds]).astype(np.float64)
    seeds = np.clip(seeds, 0.0, None)
    reset = seeds / seeds.sum() if seeds.sum() > 0 else np.full(n, 1.0 / n)

    x = reset.copy()
    for _ in range(max_iter):
        x_next = (1.0 - damping) * reset + damping * (W @ x)
        if np.abs(x_next - x).sum() < tol:
            x = x_next
            break
        x = x_next
    return x[:n_p].astype(np.float32), x[n_p:].astype(np.float32)

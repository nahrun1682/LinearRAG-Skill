# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy>=2.0",
#     "scipy>=1.14",
#     "spacy>=3.8,<3.9",
#     "sentence-transformers>=3.0",
#     "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#     "ja-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/ja_core_news_sm-3.8.0/ja_core_news_sm-3.8.0-py3-none-any.whl",
#     "en-core-web-trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl",
# ]
# ///
# NOTE: standalone script; wider Python range than the project on purpose.
"""ResiRAG: residual-query propagation on the relation-free Tri-Graph.

Research prototype layered ON TOP of the original LinearRAG retriever. The
original query_graph.py is left untouched; this module imports its helpers and
overrides only stage-1 entity activation.

Motivation (docs/research/2026-07-03-failure-analysis.md): the paper's eq.5
computes the query-sentence similarity sigma ONCE against the whole query and
reuses it every hop, so propagation has no notion of "which hop am I on."
Empirically 60% of MuSiQue failures are under-propagation (chain dies at the
seed) and ~33% are over-propagation (activation explodes), and no single delta
resolves both. ResiRAG recomputes sigma each hop from a *residual query* whose
already-covered directions have been projected out.

Design contract: with residual_strength == 0 this reduces EXACTLY to the
original activate_entities (verified in tests), so A/B comparisons are clean.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import sparse

import query_graph as qg
from common import normalize_entity


def _residual_query(q_base: np.ndarray, cover_vecs: np.ndarray,
                    strength: float) -> np.ndarray:
    """Remove the span of already-covered directions from the query.

    q_base : (d,) L2-normalized query vector.
    cover_vecs : (k, d) embeddings of the entities considered "covered" so far.
    Returns an L2-normalized residual. strength=1 -> full orthogonal projection
    removed; 0 < strength < 1 -> softer negative feedback (Rocchio-style)."""
    if strength <= 0 or cover_vecs.shape[0] == 0:
        return q_base
    # Orthonormal basis of the covered subspace, then subtract the projection.
    basis, _ = np.linalg.qr(cover_vecs.T)          # (d, r)
    proj = basis @ (basis.T @ q_base)              # component inside the span
    q_res = q_base - strength * proj
    norm = float(np.linalg.norm(q_res))
    return q_res / norm if norm > 1e-8 else q_base


def activate_entities_residual(
        a0: np.ndarray, M: sparse.csr_matrix, emb_sentences: np.ndarray,
        emb_entities: np.ndarray, query_vec: np.ndarray, *,
        delta: float = 0.8, max_iterations: int = 4, sigma_top_n: int = 200,
        residual_strength: float = 1.0, residual_k: int = 3,
        threshold_mode: str = "relative", delta_rel: float = 0.5,
        sigma_renorm: bool = True, adaptive: bool = False,
        adapt_lo: int = 2, adapt_hi: int = 20,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Eq.5 with per-hop residual-query sigma and hop-relative activation.

    threshold_mode:
      "absolute" - newly activate where score >= delta (original semantics)
      "relative" - newly activate where score >= delta_rel * max(new scores)
                   this pairs with the residual because residual queries shift
                   sigma magnitudes, making a fixed absolute delta unreliable.
    sigma_renorm: rescale each hop's sigma so its max is 1, removing the
                  magnitude drift the residual introduces (no-op at hop 1).
    adaptive: scale the residual strength by how broad the current activation
              is. The A/B showed the residual FOCUSES an over-propagating hop
              (helps over-prop) but STRIPS the seed when little is active
              (hurts under-prop). Gating strength on activation breadth applies
              it only where it helps: strength_t = residual_strength *
              clip((n_active - adapt_lo)/(adapt_hi - adapt_lo), 0, 1).
    """
    q_base = query_vec.astype(np.float32)
    q_base = q_base / (float(np.linalg.norm(q_base)) + 1e-12)

    a = a0.astype(np.float32).copy()
    levels = np.where(a > 0, 1, 0).astype(np.int32)
    trace: list[list[int]] = []

    for it in range(2, max_iterations + 2):
        # --- residual query from the top-k currently-activated entities ---
        active = np.flatnonzero(levels)
        if adaptive and active.size:
            span = max(adapt_hi - adapt_lo, 1)
            gate = min(max((active.size - adapt_lo) / span, 0.0), 1.0)
            strength_t = residual_strength * gate
        else:
            strength_t = residual_strength
        if strength_t > 0 and active.size:
            top = active[np.argsort(a[active])[::-1][:residual_k]]
            q_hop = _residual_query(q_base, emb_entities[top], strength_t)
        else:
            q_hop = q_base

        # --- sigma for this hop ---
        sigma = np.asarray(emb_sentences @ q_hop, dtype=np.float32)
        sigma = qg._sparsify_top_n(sigma, sigma_top_n)
        np.clip(sigma, 0.0, None, out=sigma)
        if sigma_renorm and strength_t > 0:
            m = float(sigma.max())
            if m > 1e-8:
                sigma = sigma / m

        # --- one propagation step (identical algebra to eq.5) ---
        spread = M.T @ (sigma * (M @ a))
        updated = np.maximum(a, spread)

        cand = (levels == 0)
        if threshold_mode == "relative":
            cand_scores = np.where(cand, updated, 0.0)
            hi = float(cand_scores.max())
            thr = delta_rel * hi if hi > 0 else np.inf
        else:
            thr = delta
        newly = cand & (updated >= thr)

        a = np.where(levels > 0, updated,
                     np.where(newly, updated, 0.0)).astype(np.float32)
        if not newly.any():
            break
        levels[newly] = it
        trace.append(np.flatnonzero(newly).tolist())
    return a, levels, trace


class ResiRetriever(qg.Retriever):
    """LinearRAG retriever whose stage-1 activation uses residual-query sigma.

    Stage 2 (passage seeds + PPR) is unchanged and still uses the full original
    query, so this isolates the effect of hop-aware activation."""

    def __call__(self, query: str, top_k: int = 5, delta: float = 0.8,
                 max_iterations: int = 4, lam: float = 1.5, w_p: float = 0.05,
                 damping: float = 0.5, sigma_top_n: int = 200,
                 residual_strength: float = 1.0, residual_k: int = 3,
                 threshold_mode: str = "relative", delta_rel: float = 0.5,
                 sigma_renorm: bool = True, adaptive: bool = False,
                 adapt_lo: int = 2, adapt_hi: int = 20) -> dict:
        index = self.index
        query_vec = self.embed([query])[0]

        q_entities = [normalize_entity(ent.text) for ent in self.nlp(query).ents
                      if not qg._is_numeric_label(ent.label_)]
        q_entities = [e for e in q_entities if e]
        q_vecs = (self.embed(q_entities) if q_entities
                  else np.zeros((0, 1), dtype=np.float32))
        a0 = qg.initial_activation(q_vecs, index.emb_entities)

        if residual_strength <= 0 and threshold_mode == "absolute":
            # exact original path (full-query sigma + absolute delta)
            sigma = qg._sparsify_top_n(
                np.asarray(index.emb_sentences @ query_vec), sigma_top_n)
            activation, levels, trace = qg.activate_entities(
                a0, index.M, sigma, delta=delta, max_iterations=max_iterations)
        else:
            # residual_strength==0 with relative mode isolates C2 (relative
            # threshold) alone: sigma stays the full-query sigma, only the
            # activation rule changes.
            activation, levels, trace = activate_entities_residual(
                a0, index.M, index.emb_sentences, index.emb_entities, query_vec,
                delta=delta, max_iterations=max_iterations,
                sigma_top_n=sigma_top_n, residual_strength=residual_strength,
                residual_k=residual_k, threshold_mode=threshold_mode,
                delta_rel=delta_rel, sigma_renorm=sigma_renorm,
                adaptive=adaptive, adapt_lo=adapt_lo, adapt_hi=adapt_hi)

        sim_qp = index.emb_passages @ query_vec
        p_seeds = qg.passage_seed_scores(sim_qp, index.C, activation, levels,
                                         lam=lam, w_p=w_p)
        p_scores, _ = qg._ppr_iterate(self._W, self._n_p, p_seeds, activation,
                                      damping=damping)

        order = np.argsort(p_scores)[::-1][:top_k]
        activated = [
            {"entity": index.entities[i], "score": round(float(activation[i]), 4),
             "level": int(levels[i])}
            for i in np.flatnonzero(levels)]
        activated.sort(key=lambda r: (r["level"], -r["score"]))
        return {
            "query": query,
            "query_entities": q_entities,
            "activated_entities": activated,
            "passages": [
                {"rank": r + 1, "id": index.passages[i]["id"],
                 "title": index.passages[i]["title"],
                 "score": round(float(p_scores[i]), 6),
                 "text": index.passages[i]["text"]}
                for r, i in enumerate(order)],
            "params": {"top_k": top_k, "delta": delta, "lam": lam, "w_p": w_p,
                       "damping": damping, "max_iterations": max_iterations,
                       "sigma_top_n": sigma_top_n,
                       "residual_strength": residual_strength,
                       "residual_k": residual_k, "threshold_mode": threshold_mode,
                       "delta_rel": delta_rel, "sigma_renorm": sigma_renorm},
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="ResiRAG residual-query retrieval")
    ap.add_argument("--index", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--residual-strength", type=float, default=1.0)
    ap.add_argument("--residual-k", type=int, default=3)
    ap.add_argument("--threshold-mode", default="relative",
                    choices=["relative", "absolute"])
    ap.add_argument("--delta-rel", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=0.8)
    args = ap.parse_args()

    from common import Embedder, TriGraphIndex, load_nlp
    index = TriGraphIndex.load(args.index)
    retr = ResiRetriever(index, Embedder(index.meta["embedding_model"]),
                         load_nlp(index.meta["spacy_model"]))
    res = retr(args.query, top_k=args.top_k,
               residual_strength=args.residual_strength,
               residual_k=args.residual_k, threshold_mode=args.threshold_mode,
               delta_rel=args.delta_rel, delta=args.delta)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

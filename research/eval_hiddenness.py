# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy>=2.0","scipy>=1.14","spacy>=3.8,<3.9","sentence-transformers>=3.0","rank-bm25>=0.2.2",
#     "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#     "ja-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/ja_core_news_sm-3.8.0/ja_core_news_sm-3.8.0-py3-none-any.whl",
#     "en-core-web-trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl",
# ]
# ///
"""The 'hidden hop' diagnosis (experiments A + B).

Hypothesis: a 22M cross-encoder matched GPT-4o because the HARD, query-invisible
2nd-hop gold ('Ermengarde died in 851' — no 'Lothair'/'mother' in it) is either
(X) absent from the BM25 candidate pool, or (Y) missed by everyone. We quantify
'hiddenness' of each gold passage w.r.t. the query and stratify by it.

hiddenness(gold) = 1 - percentile-rank of the gold's query-similarity among the
    corpus (lexical BM25 and dense cosine, averaged). High = the gold looks
    unlike the query = only reachable by reading another hop or a graph bridge.
Per question we take the MOST hidden gold (the binding constraint for AllGold).

Experiment A: for each hiddenness bin, report AllGoldHit of bm25 / coverSoft /
    crossenc / gpt4o (gpt4o read from cache). Do set-aware methods hold up as
    hiddenness rises while pointwise crossenc collapses?
Experiment B: pool coverage — is the hidden gold even in BM25 top-50? And does
    the GRAPH (PPR) pool reach hidden golds that BM25 misses?
"""
import argparse
import json
import re
import sys

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, ".claude/skills/linearrag/scripts")
sys.path.insert(0, "research")
from common import Embedder, TriGraphIndex, load_nlp  # noqa: E402
from resirag import ResiRetriever  # noqa: E402
from stage2_experiment import _minmax, select_cover_soft  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)
ap.add_argument("--corpus", required=True)
ap.add_argument("--questions", required=True)
ap.add_argument("--top-n", type=int, default=50)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--limit", type=int, default=300)
ap.add_argument("--gamma", type=float, default=0.5)
ap.add_argument("--ce-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
ap.add_argument("--llm-cache", default=None, help="gpt4o raw-selection cache json")
a = ap.parse_args()

_tok = re.compile(r"\w+", re.UNICODE)
tok = lambda s: _tok.findall(s.lower())
corpus = [json.loads(l) for l in open(a.corpus, encoding="utf-8")]
ids = [p["id"] for p in corpus]
id2row = {p["id"]: i for i, p in enumerate(corpus)}
index = TriGraphIndex.load(a.index)
assert all(corpus[i]["id"] == index.passages[i]["id"] for i in range(0, len(corpus), 500))
bm25 = BM25Okapi([tok(p.get("title", "") + " " + p["text"]) for p in corpus])
embed = Embedder(index.meta["embedding_model"])
retr = ResiRetriever(index, embed, load_nlp(index.meta["spacy_model"]))
C = index.C.tocsr()
embP = np.asarray(index.emb_passages)
qs = json.load(open(a.questions, encoding="utf-8"))[: a.limit]
llm_cache = json.load(open(a.llm_cache, encoding="utf-8")) if a.llm_cache else {}

from sentence_transformers import CrossEncoder
ce = CrossEncoder(a.ce_model, max_length=512)


def clip(t, w=160):
    return " ".join(t.split()[:w])


def coversoft(cd, rows):
    ent = [C[r].indices for r in rows]
    w = cd["activation"] / np.maximum(cd["levels"], 1)
    df = np.zeros_like(w)
    for e in ent:
        df[e] += 1.0
    w_idf = w * np.log(1.0 + len(ent) / np.maximum(df, 1.0))
    w_idf = w_idf / (max((float(w_idf[e].sum()) for e in ent), default=0.0) or 1.0)
    dn = _minmax(embP[rows] @ cd["query_vec"])
    return [rows[i] for i in select_cover_soft(ent, dn, w_idf, a.k, 1.0, a.gamma)]


rows_per_q = []
for q in qs:
    gold = set(q.get("gold_ids", []))
    if not gold:
        continue
    qv = embed([q["question"]])[0]
    # dense-sim percentile of each gold vs the whole corpus
    dcorpus = embP @ qv
    bm_scores_all = bm25.get_scores(tok(q["question"]))
    cd = retr.stage2_candidates(q["question"], top_n=a.top_n)
    bm_rows = list(np.argsort(bm_scores_all)[::-1][:a.top_n])
    ppr_rows = list(cd["cand_rows"])

    def hidden(gid):
        r = id2row[gid]
        dpr = (dcorpus < dcorpus[r]).mean()          # dense percentile
        lpr = (bm_scores_all < bm_scores_all[r]).mean()  # lexical percentile
        return 1.0 - 0.5 * (dpr + lpr)               # high = hidden
    gold_h = {g: hidden(g) for g in gold}
    worst = max(gold_h.values())  # most hidden gold = binding for AllGold

    # selections
    bm_pick = {ids[r] for r in bm_rows[:a.k]}
    cs_pick = {ids[r] for r in coversoft(cd, bm_rows)}
    pairs = [[q["question"], corpus[r].get("title", "") + ". " + clip(corpus[r]["text"])]
             for r in bm_rows]
    ce_order = np.argsort(ce.predict(pairs))[::-1][:a.k]
    ce_pick = {ids[bm_rows[i]] for i in ce_order}
    llm_sel = llm_cache.get(q["question"])
    llm_pick = set(llm_sel[:a.k]) if llm_sel else None

    rows_per_q.append({
        "gold": gold, "worst_h": worst,
        "in_bm_pool": gold <= {ids[r] for r in bm_rows},
        "in_ppr_pool": gold <= {ids[r] for r in ppr_rows},
        "in_union_pool": gold <= ({ids[r] for r in bm_rows} | {ids[r] for r in ppr_rows}),
        "bm25": gold <= bm_pick, "coverSoft": gold <= cs_pick,
        "crossenc": gold <= ce_pick,
        "gpt4o": (gold <= llm_pick) if llm_pick is not None else None,
    })

n = len(rows_per_q)
print(f"index={a.index}  n={n}")

# --- Experiment A: AllGoldHit vs hiddenness bin ---
hs = np.array([r["worst_h"] for r in rows_per_q])
edges = np.quantile(hs, [0, 0.25, 0.5, 0.75, 1.0])
print("\n=== A: AllGoldHit@5 by hiddenness quartile (of most-hidden gold) ===")
print(f"{'bin(worst_h)':22s} {'n':>4s} {'bm25':>6s} {'coverSoft':>9s} {'crossenc':>9s} {'gpt4o':>6s}")
for b in range(4):
    lo, hi = edges[b], edges[b + 1]
    sel = [r for r in rows_per_q if (lo <= r["worst_h"] <= hi if b == 3 else lo <= r["worst_h"] < hi)]
    if not sel:
        continue
    def rate(key):
        v = [r[key] for r in sel if r[key] is not None]
        return f"{np.mean(v):.0%}" if v else "  -"
    print(f"[{lo:.2f},{hi:.2f}]{'':10s} {len(sel):>4d} {rate('bm25'):>6s} "
          f"{rate('coverSoft'):>9s} {rate('crossenc'):>9s} {rate('gpt4o'):>6s}")

# --- Experiment B: pool coverage vs hiddenness ---
print("\n=== B: all-gold IN POOL by hiddenness quartile ===")
print(f"{'bin(worst_h)':22s} {'n':>4s} {'bm25pool':>9s} {'pprpool':>8s} {'union':>6s}")
for b in range(4):
    lo, hi = edges[b], edges[b + 1]
    sel = [r for r in rows_per_q if (lo <= r["worst_h"] <= hi if b == 3 else lo <= r["worst_h"] < hi)]
    if not sel:
        continue
    print(f"[{lo:.2f},{hi:.2f}]{'':10s} {len(sel):>4d} "
          f"{np.mean([r['in_bm_pool'] for r in sel]):>9.0%} "
          f"{np.mean([r['in_ppr_pool'] for r in sel]):>8.0%} "
          f"{np.mean([r['in_union_pool'] for r in sel]):>6.0%}")

print(f"\noverall pool recall(all-gold): bm25={np.mean([r['in_bm_pool'] for r in rows_per_q]):.0%} "
      f"ppr={np.mean([r['in_ppr_pool'] for r in rows_per_q]):.0%} "
      f"union={np.mean([r['in_union_pool'] for r in rows_per_q]):.0%}")

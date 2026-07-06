# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy>=2.0","scipy>=1.14","spacy>=3.8,<3.9","sentence-transformers>=3.0","rank-bm25>=0.2.2","openai>=1.40",
#     "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#     "ja-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/ja_core_news_sm-3.8.0/ja_core_news_sm-3.8.0-py3-none-any.whl",
#     "en-core-web-trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl",
# ]
# ///
"""SPRIG head-to-head: same token-free NER+PPR graph, used as RETRIEVER (SPRIG)
vs as INJECTOR (ours).

SPRIG (arXiv:2602.23372) uses the graph as a *retriever*: seed PPR (from query
entities / BM25 / dense / RRF) and RANK passages by PPR score, take top-k. We use
the same graph as an *injector*: keep a BM25 pool and ADD graph-PPR candidates,
leaving selection to a later stage.

Hypothesis: PPR-ranking (any ranking) still pushes query-dissimilar hidden hops
DOWN, so SPRIG's top-N can bury them; injection ADDS them regardless of rank. So
at the pool level (all-gold in top-N), especially in hidden bins, injection >
graph-as-retriever. We reproduce SPRIG's seeding+PPR-ranking on OUR graph (fair:
same graph, different usage) and compare pool recall@N + AllGoldHit@5.

Methods (each yields a ranked top-N passage list):
  bm25            BM25 top-N
  rrf             RRF(BM25, dense) top-N            (SPRIG's non-graph strong base)
  sprigGraph      PPR seeded by query entities, PPR-ranked
  sprigHybrid     PPR seeded by BM25 passages, PPR-ranked
  sprigRRF        PPR seeded by RRF scores, PPR-ranked   (SPRIG's best variant)
  ours            BM25 top(base) UNION graphPPR-injected top(add)   (injector)
"""
import argparse
import json
import re
import sys

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, ".claude/skills/linearrag/scripts")
sys.path.insert(0, "research")
from common import Embedder, TriGraphIndex, load_nlp, normalize_entity  # noqa: E402
from resirag import ResiRetriever  # noqa: E402
import query_graph as qg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)
ap.add_argument("--corpus", required=True)
ap.add_argument("--questions", required=True)
ap.add_argument("--pool-n", type=int, default=50, help="pool depth for recall@N")
ap.add_argument("--base-n", type=int, default=30)
ap.add_argument("--add-n", type=int, default=20)
ap.add_argument("--limit", type=int, default=300)
ap.add_argument("--damping", type=float, default=0.5)
ap.add_argument("--llm", action="store_true", help="also run LLM selection on a pool")
ap.add_argument("--llm-pool", default="both", choices=["both", "fuseRRF", "bm25"],
                help="which candidate pool(s) the LLM selects from (isolates pool value)")
ap.add_argument("--model", default="gpt-5.4-mini")
ap.add_argument("--workers", type=int, default=8)
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
nlp = load_nlp(index.meta["spacy_model"])
retr = ResiRetriever(index, embed, nlp)
embP = np.asarray(index.emb_passages)
embE = np.asarray(index.emb_entities)
W, n_p = retr._W, retr._n_p
n_e = embE.shape[0]
qs = json.load(open(a.questions, encoding="utf-8"))[: a.limit]


def ppr_rank(passage_seeds, entity_seeds):
    ps, _ = qg._ppr_iterate(W, n_p, passage_seeds, entity_seeds, damping=a.damping)
    return list(np.argsort(ps)[::-1])


def rrf_order(order_a, order_b, kk=60):
    sc = {}
    for order in (order_a, order_b):
        for rank, r in enumerate(order):
            sc[r] = sc.get(r, 0.0) + 1.0 / (kk + rank + 1)
    return sorted(sc, key=lambda r: -sc[r])


def dedup(seq, n):
    out, seen = [], set()
    for r in seq:
        if r not in seen:
            seen.add(r); out.append(r)
        if len(out) >= n:
            break
    return out


def rrf_multi(orders, kk=60, depth=200):
    sc = {}
    for order in orders:
        for rank, r in enumerate(order[:depth]):
            sc[r] = sc.get(r, 0.0) + 1.0 / (kk + rank + 1)
    return sorted(sc, key=lambda r: -sc[r])


def interleave(vis, hid, n_vis, n_hid):
    """Regime slotting: n_vis unique from the visible list, then n_hid unique
    NEW ones from the hidden list. Slots, not score fusion."""
    out = dedup(list(vis), n_vis)
    seen = set(out)
    for r in hid:
        if len(out) >= n_vis + n_hid:
            break
        if r not in seen:
            seen.add(r); out.append(r)
    return out


def team_draft(vis, hid, n):
    """Parameter-free alternation (team-draft interleaving): fair round-robin
    draft between the two ranked lists, skipping already-picked. No split ratio
    to tune. Self-adapts: when the lists agree (visible-only queries) the visible
    list is effectively drafted deeper; when they diverge each regime gets half."""
    out, seen = [], set()
    ia = ib = 0
    turn_vis = True
    while len(out) < n and (ia < len(vis) or ib < len(hid)):
        if turn_vis:
            while ia < len(vis) and vis[ia] in seen:
                ia += 1
            if ia < len(vis):
                seen.add(vis[ia]); out.append(vis[ia]); ia += 1
        else:
            while ib < len(hid) and hid[ib] in seen:
                ib += 1
            if ib < len(hid):
                seen.add(hid[ib]); out.append(hid[ib]); ib += 1
        turn_vis = not turn_vis
    return out


def min_rank_merge(vis, hid, n):
    """Parameter-free merge: each candidate's score is its BEST (minimum) rank
    across the two lists, not the sum (RRF sums, which buries a hidden hop that
    ranks terribly in the visible list). Tie-break by the other rank."""
    rv = {r: i for i, r in enumerate(vis)}
    rh = {r: i for i, r in enumerate(hid)}
    BIG = 10 ** 9
    cand = set(rv) | set(rh)
    def key(r):
        a_, b_ = rv.get(r, BIG), rh.get(r, BIG)
        return (min(a_, b_), max(a_, b_))
    return sorted(cand, key=key)[:n]


methods = ["bm25", "rrf", "sprigGraph", "sprigHybrid", "sprigRRF", "ours",
           "fuseBM25", "fuseRRF", "rrf3way", "slot32", "slot23", "teamdraft", "minrank"]
rows = []
retr.stage2_candidates(qs[0]["question"], top_n=a.pool_n)
for q in qs:
    gold = set(q.get("gold_ids", []))
    if not gold:
        continue
    Q = q["question"]
    qv = embed([Q])[0]
    bm_scores = bm25.get_scores(tok(Q))
    bm_order = list(np.argsort(bm_scores)[::-1])
    dense_scores = embP @ qv
    dense_order = list(np.argsort(dense_scores)[::-1])
    rrf_ord = rrf_order(bm_order, dense_order)

    # entity seeds from query entities (SPRIG-Graph)
    q_ents = [normalize_entity(e.text) for e in nlp(Q).ents
              if not qg._is_numeric_label(e.label_)]
    q_ents = [e for e in q_ents if e]
    eseed = np.zeros(n_e)
    if q_ents:
        sims = embed(q_ents) @ embE.T
        for j in range(sims.shape[0]):
            ge = int(np.argmax(sims[j]))
            eseed[ge] = max(eseed[ge], float(sims[j, ge]))
    # passage seeds: BM25 top and RRF top (normalized)
    pseed_bm = np.zeros(n_p)
    for rnk, r in enumerate(bm_order[:a.pool_n]):
        pseed_bm[r] = max(0.0, float(bm_scores[r]))
    if pseed_bm.sum() > 0:
        pseed_bm /= pseed_bm.sum()
    pseed_rrf = np.zeros(n_p)
    for rnk, r in enumerate(rrf_ord[:a.pool_n]):
        pseed_rrf[r] = 1.0 / (60 + rnk + 1)
    if pseed_rrf.sum() > 0:
        pseed_rrf /= pseed_rrf.sum()

    # our injector pool: BM25 base UNION graphPPR-injected
    cd = retr.stage2_candidates(Q, top_n=a.base_n + a.add_n)
    base = bm_order[:a.base_n]
    ppr_inj = [int(r) for r in cd["cand_rows"] if int(r) not in set(base)][:a.add_n]

    # pure entity-seeded PPR ranking (= sprigGraph); reuse for fusion injection
    eppr = ppr_rank(np.zeros(n_p), eseed)
    rrf3 = rrf_multi([bm_order, dense_order, eppr])
    base_bm = bm_order[:a.base_n]
    base_rrf = rrf_ord[:a.base_n]
    inj_bm = [r for r in eppr if r not in set(base_bm)][:a.add_n]
    inj_rrf = [r for r in eppr if r not in set(base_rrf)][:a.add_n]
    pools = {
        "bm25": bm_order[:a.pool_n],
        "rrf": rrf_ord[:a.pool_n],
        "sprigGraph": eppr[:a.pool_n],
        "sprigHybrid": ppr_rank(pseed_bm, np.zeros(n_e))[:a.pool_n],
        "sprigRRF": ppr_rank(pseed_rrf, np.zeros(n_e))[:a.pool_n],
        "ours": dedup(list(base) + ppr_inj, a.pool_n),  # OLD: BM25 + LinearRAG-PPR(DPR-mixed)
        # NEW fusion: visible base (BM25/RRF) UNION pure entity-PPR (hidden)
        "fuseBM25": dedup(list(base_bm) + inj_bm, a.pool_n),
        "fuseRRF": dedup(list(base_rrf) + inj_rrf, a.pool_n),
        # token-free SELECTION contenders (what matters is their top-5):
        # rrf3way: naive score fusion of all 3 signals — predicted to bury
        #          hidden hops (they rank terribly in the two query-based lists)
        "rrf3way": rrf3[:a.pool_n],
        # slotN M: regime slotting — top-5 = N visible slots (RRF) + M hidden
        #          slots (entity-PPR), dedup; rest of pool = fuseRRF order
        "slot32": dedup(interleave(base_rrf, eppr if q_ents else [], 3, 2)
                        + list(base_rrf) + inj_rrf, a.pool_n),
        "slot23": dedup(interleave(base_rrf, eppr if q_ents else [], 2, 3)
                        + list(base_rrf) + inj_rrf, a.pool_n),
        # parameter-free: fair round-robin draft between RRF and entity-PPR
        "teamdraft": dedup(team_draft(rrf_ord, eppr if q_ents else [], a.pool_n)
                           + inj_rrf, a.pool_n),
        # parameter-free: enter at your BEST rank in either regime (min, not sum
        # like RRF) so a hidden hop ranked #1 by entPPR is not buried by its
        # terrible RRF rank. Symmetric, no split ratio, no starting turn.
        "minrank": min_rank_merge(rrf_ord, eppr if q_ents else [], a.pool_n),
    }
    dcorpus = dense_scores

    def hidden(gid):
        r = id2row[gid]
        return 1.0 - 0.5 * ((dcorpus < dcorpus[r]).mean() + (bm_scores < bm_scores[r]).mean())
    worst_h = max(hidden(g) for g in gold)
    rec = {"worst_h": worst_h, "q": Q, "gold": gold,
           "fuseRRF_rows": list(pools["fuseRRF"]),  # LLM selects from SAME pool as min-rank
           "bm25_rows": list(pools["bm25"])}        # ...vs the plain BM25 pool (pool-value test)
    for m, pool in pools.items():
        pset = {ids[r] for r in pool}
        rec[m + "_pool"] = gold <= pset
        rec[m + "_ag5"] = gold <= {ids[r] for r in pool[:5]}
    rows.append(rec)

n = len(rows)
print(f"index={a.index}  n={n}  pool@{a.pool_n}")
print("\n=== all-gold POOL recall @%d (overall) ===" % a.pool_n)
for m in methods:
    print(f"  {m:12s} {np.mean([r[m+'_pool'] for r in rows]):>6.1%}")
print("\n=== AllGoldHit@5 (each method's own top-5) ===")
for m in methods:
    print(f"  {m:12s} {np.mean([r[m+'_ag5'] for r in rows]):>6.1%}")
print("\n=== POOL recall @%d by hiddenness quartile ===" % a.pool_n)
hs = np.array([r["worst_h"] for r in rows])
edges = np.quantile(hs, [0, .25, .5, .75, 1.0])
print(f"{'bin':16s} {'n':>4s} " + " ".join(f"{m:>11s}" for m in methods))
for b in range(4):
    lo, hi = edges[b], edges[b + 1]
    sel = [r for r in rows if (lo <= r["worst_h"] <= hi if b == 3 else lo <= r["worst_h"] < hi)]
    if not sel:
        continue
    cells = " ".join(f"{np.mean([r[m+'_pool'] for r in sel]):>10.0%} " for m in methods)
    print(f"[{lo:.2f},{hi:.2f}]{'':3s} {len(sel):>4d} {cells}")

# === LLM reference: select from the SAME fuseRRF pool (vs token-free min-rank) ===
if a.llm:
    import concurrent.futures as cf
    import os
    import time
    from openai import OpenAI

    def key():
        for l in open(".env"):
            if l.strip().startswith("OPENAI_API_KEY"):
                return l.split("=", 1)[1].strip().strip('"').strip("'")
    client = OpenAI(api_key=key())
    sel_re = re.compile(r"\[(\d+)\]")
    is_reasoning = not a.model.startswith("gpt-4o")  # gpt-5.x/o-series: no temp=0
    PROMPT = ("I will provide you with {num} passages, each indicated by a numerical "
              "identifier []. Select the passages based on their relevance to the search "
              "query: {q}.\n\n{ctx}\n\nSearch Query: {q}\n\nPlease follow the steps below:\n"
              "Step 1. List the information requirements to answer the query.\n"
              "Step 2. For each requirement, find the passages that hold that information.\n"
              "Step 3. Choose the passages that together cover all requirements. "
              "Number of passages is unlimited. Final line format: "
              "'### Final Selection: [] []', e.g., ### Final Selection: [2] [1].")
    name = a.corpus.split("/")[-2]

    def clip(t, w=110):
        return " ".join(t.split()[:w])

    def gpt_pick(item, pool, cache):
        if item["q"] in cache and cache[item["q"]] is not None:
            picks = cache[item["q"]]
        else:
            prows = item[pool + "_rows"]
            ctx = "\n".join(f"[{j+1}] ({corpus[r].get('title','')}) {clip(corpus[r]['text'])}"
                            for j, r in enumerate(prows))
            prompt = PROMPT.format(num=len(prows), q=item["q"], ctx=ctx)
            picks = None
            for att in range(5):
                try:
                    kw = dict(model=a.model, messages=[{"role": "user", "content": prompt}])
                    if not is_reasoning:
                        kw["temperature"] = 0
                    r = client.chat.completions.create(**kw)
                    after = r.choices[0].message.content.split("Final Selection")[-1]
                    picks = []
                    for p in [int(x) for x in sel_re.findall(after)]:
                        if 1 <= p <= len(prows) and ids[prows[p-1]] not in picks:
                            picks.append(ids[prows[p-1]])
                    break
                except Exception as e:
                    if att == 4:
                        print("  ! give up:", str(e)[:80]); picks = None
                    else:
                        time.sleep(2 ** att)
        cache[item["q"]] = picks
        return picks

    llm_pools = ["bm25", "fuseRRF"] if a.llm_pool == "both" else [a.llm_pool]
    print(f"\n=== LLM pool-value ({a.model}) — AllGoldHit@5 ===")
    for pool in llm_pools:
        cfile = f"research/cache_llmref_{name}_{a.model}_{pool}_{a.limit}.json"
        cache = json.load(open(cfile, encoding="utf-8")) if os.path.exists(cfile) else {}
        with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
            picks = list(ex.map(lambda it: gpt_pick(it, pool, cache), rows))
        json.dump(cache, open(cfile, "w", encoding="utf-8"), ensure_ascii=False)
        hit = np.mean([it["gold"] <= set((p or [])[:5]) for it, p in zip(rows, picks)])
        nfail = sum(1 for p in picks if p is None)
        print(f"  {a.model} @ {pool:8s} {hit:>6.1%}   (fails={nfail}/{len(rows)})")
    print("  (compare token-free min-rank/slot on the fuseRRF pool, above)")

# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy>=2.0","scipy>=1.14","spacy>=3.8,<3.9","sentence-transformers>=3.0","rank-bm25>=0.2.2","openai>=1.40",
#     "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#     "ja-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/ja_core_news_sm-3.8.0/ja_core_news_sm-3.8.0-py3-none-any.whl",
#     "en-core-web-trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl",
# ]
# ///
"""LLM set-selection UPPER BOUND vs coverSoft (token-free) on the SAME candidates.

Bridges our eval and SETR's world: same BM25 top-N candidate pool, same
gold-passage metric, three selectors:
  bm25          - top-k by BM25 score (no selection)
  bm25+coverSoft- our token-free selection (graph-activation coverage)
  bm25+gpt4o    - SETR-style LLM set selection (the paid upper bound)
The headline we want: coverSoft recovers X% of the LLM-selection gain at 0 tokens.
Only the upper-bound baseline uses an LLM; the method stays token-free.
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, ".claude/skills/linearrag/scripts")
sys.path.insert(0, "research")
from common import Embedder, TriGraphIndex, load_nlp  # noqa: E402
from resirag import ResiRetriever  # noqa: E402
from stage2_experiment import _minmax, select_cover_soft  # noqa: E402


def load_env_key(path=".env"):
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith("OPENAI_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OPENAI_API_KEY not found in .env")


ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)
ap.add_argument("--corpus", required=True)
ap.add_argument("--questions", required=True)
ap.add_argument("--top-n", type=int, default=50)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--gamma", type=float, default=0.5)
ap.add_argument("--limit", type=int, default=200)
ap.add_argument("--model", default="gpt-4o-2024-08-06")
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--words", type=int, default=110, help="truncate passage words")
a = ap.parse_args()

_tok = re.compile(r"\w+", re.UNICODE)
tok = lambda s: _tok.findall(s.lower())

corpus = [json.loads(l) for l in open(a.corpus, encoding="utf-8")]
ids = [p["id"] for p in corpus]
index = TriGraphIndex.load(a.index)
assert len(corpus) == len(index.passages) and all(
    corpus[i]["id"] == index.passages[i]["id"] for i in range(0, len(corpus), 500))
bm25 = BM25Okapi([tok(p.get("title", "") + " " + p["text"]) for p in corpus])
embed = Embedder(index.meta["embedding_model"])
retr = ResiRetriever(index, embed, load_nlp(index.meta["spacy_model"]))
C = index.C.tocsr()
embP = np.asarray(index.emb_passages)
qs = json.load(open(a.questions, encoding="utf-8"))[: a.limit]

# Verbatim from SETR (Lee et al. 2025, Figure 2) — kept word-for-word (incl.
# their grammar) so the upper bound faithfully reproduces their teacher prompt.
PROMPT = """I will provide you with {num} passages, each indicated by a numerical identifier []. Select the passages based on their relevance to the search query: {q}.

{ctx}

Search Query: {q}

Please follow the steps below:
Step 1. Please list up the information requirements to answer the query.
Step 2. for each requirement in Step 1, find the passages that has the information of the requirement.
Step 3. Choose the passages that mostly covers clear and diverse information to answer the query. Number of passages is unlimited. The format of final output should be '### Final Selection: [] []', e.g., ### Final Selection: [2] [1]."""


def clip(text, w):
    return " ".join(text.split()[:w])


def coversoft_pick(cd, rows):
    ent = [C[r].indices for r in rows]
    w = cd["activation"] / np.maximum(cd["levels"], 1)
    df = np.zeros_like(w)
    for e_arr in ent:
        df[e_arr] += 1.0
    idf = np.log(1.0 + len(ent) / np.maximum(df, 1.0))
    w_idf = w * idf
    w_idf = w_idf / (max((float(w_idf[e].sum()) for e in ent), default=0.0) or 1.0)
    dn = _minmax(embP[rows] @ cd["query_vec"])
    return [rows[i] for i in select_cover_soft(ent, dn, w_idf, a.k, 1.0, a.gamma)]


# ---- pass 1: candidates + coverSoft (sequential, graph NER) ----
items = []
retr.stage2_candidates(qs[0]["question"], top_n=a.top_n)
for q in qs:
    gold = set(q.get("gold_ids", []))
    if not gold:
        continue
    cd = retr.stage2_candidates(q["question"], top_n=a.top_n)
    bm_rows = list(np.argsort(bm25.get_scores(tok(q["question"])))[::-1][:a.top_n])
    items.append({
        "q": q["question"], "gold": gold, "bm_rows": bm_rows,
        "bm25": [ids[r] for r in bm_rows[:a.k]],
        "coversoft": [ids[r] for r in coversoft_pick(cd, bm_rows)],
    })
print(f"pass1 done: {len(items)} questions with gold")

# ---- pass 2: GPT-4o selection (concurrent) ----
from openai import OpenAI
client = OpenAI(api_key=load_env_key())
sel_re = re.compile(r"\[(\d+)\]")


name = a.corpus.split("/")[-2]
CACHE = f"research/cache_llm_{name}_{a.model}_{a.limit}.json"
cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}


def llm_select(it):
    """Return the RAW llm-selected id list (no padding). None on total failure."""
    if it["q"] in cache:
        return cache[it["q"]]
    ctx = "\n".join(f"[{j+1}] ({corpus[r].get('title','')}) {clip(corpus[r]['text'], a.words)}"
                    for j, r in enumerate(it["bm_rows"]))
    prompt = PROMPT.format(num=len(it["bm_rows"]), q=it["q"], ctx=ctx)
    for attempt in range(5):
        try:
            r = client.chat.completions.create(
                model=a.model, temperature=0,
                messages=[{"role": "user", "content": prompt}])
            txt = r.choices[0].message.content
            after = txt.split("Final Selection")[-1]
            picks = [int(x) for x in sel_re.findall(after)]
            rows = []
            for p in picks:
                if 1 <= p <= len(it["bm_rows"]) and it["bm_rows"][p - 1] not in rows:
                    rows.append(it["bm_rows"][p - 1])
            return [ids[r] for r in rows]
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                print("  llm fail:", str(e)[:80])
                return None
            time.sleep(2 ** attempt)


t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
    llm_raw = list(ex.map(llm_select, items))
print(f"pass2 (llm) done in {time.time()-t0:.0f}s  "
      f"(failures: {sum(x is None for x in llm_raw)})")
cache = {it["q"]: sel for it, sel in zip(items, llm_raw) if sel is not None}
json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
print(f"cached raw selections -> {CACHE}")


def pad(it, sel):
    out = list(sel)
    for r in it["bm_rows"]:
        if len(out) >= a.k:
            break
        if ids[r] not in out:
            out.append(ids[r])
    return out[:a.k]


def score(picklist):
    rec = ag = 0
    for it, pid in zip(items, picklist):
        got = len(it["gold"] & set(pid))
        rec += got / len(it["gold"])
        ag += (got == len(it["gold"]))
    n = len(items)
    return rec / n, ag / n


ok = [(it, sel) for it, sel in zip(items, llm_raw) if sel is not None]
n_sel = [len(sel) for _, sel in ok]
print(f"\n{name}  n={len(items)}  model={a.model}  "
      f"llm avg#selected={sum(n_sel)/max(len(n_sel),1):.2f}")
print(f"{'selector':22s} {'GoldRecall@5':>13s} {'AllGoldHit@5':>13s}")
rows_out = [
    ("bm25", [it["bm25"] for it in items]),
    ("bm25+coverSoft", [it["coversoft"] for it in items]),
    ("bm25+gpt4o(raw)", [sel[:a.k] if sel is not None else it["bm25"]
                         for it, sel in zip(items, llm_raw)]),
    ("bm25+gpt4o(+pad)", [pad(it, sel) if sel is not None else it["bm25"]
                          for it, sel in zip(items, llm_raw)]),
]
for tag, pl in rows_out:
    r, g = score(pl)
    print(f"{tag:22s} {r:>13.1%} {g:>13.1%}")

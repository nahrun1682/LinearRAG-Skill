# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy>=2.0","scipy>=1.14","spacy>=3.8,<3.9","sentence-transformers>=3.0",
#     "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#     "ja-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/ja_core_news_sm-3.8.0/ja_core_news_sm-3.8.0-py3-none-any.whl",
#     "en-core-web-trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl",
# ]
# ///
"""Gold-passage retrieval metrics for the LinearRAG improvement study.

Adds gold-passage recall alongside the answer-string Contain@5 the project has
used so far. Contain@5 is confounded by paraphrase (false negatives) and common
answer strings like dates (false positives); gold-passage recall measures
whether retrieval actually reached the annotated supporting passages, which is
the metric PropRAG / HippoRAG2 / BridgeRAG report and the one that shows the
multi-hop bridge either being traversed or not.

Gold identification strategies (per dataset annotation shape):
  ids      - gold = explicit passage ids in the question's "gold_ids" field.
             Produced by research/build_gold_datasets.py from the canonical
             HippoRAG-format corpora; the ROBUST option (passage-level, handles
             musique's same-title multi-para hops). Preferred for 2wiki/hotpot/
             musique *_hpr datasets.
  title    - gold = passages whose title matches an evidence entity/document
             title. Used for JEMHopQA (evidence = (subj, rel, obj)) where
             subjects are Wikipedia page titles.
  sentence - gold = passages whose text contains an annotated supporting
             sentence (fragile under aggressive chunking; kept for reference).

Metrics reported (@k):
  Contain@k        answer string present in any top-k passage (legacy)
  GoldTitleRecall  mean over questions of |gold hit in top-k| / |gold|
  AllGoldHit@k     fraction of questions with ALL gold units in top-k (the
                   strict multi-hop measure: every bridge passage retrieved)
"""
from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, ".claude/skills/linearrag/scripts")
from common import Embedder, TriGraphIndex, load_nlp  # noqa: E402
from resirag import ResiRetriever  # noqa: E402


def norm(s: str) -> str:
    return s.casefold().strip()


def gold_units_title(q: dict, corpus_titles: set[str]) -> set[str]:
    """Gold = evidence subjects (and objects if they name a page) that exist as
    passage titles. Returns the set of gold TITLES."""
    ev = q.get("evidence") or []
    subjects = set()
    for e in ev:
        if isinstance(e, (list, tuple)) and e:
            if e[0] in corpus_titles:
                subjects.add(e[0])
            # object may also be a page (e.g. bridged entity), include if titled
            if len(e) >= 3 and isinstance(e[2], str) and e[2] in corpus_titles:
                subjects.add(e[2])
    return subjects


def gold_units_sentence(q: dict) -> set[str]:
    """Gold = the set of supporting sentences (normalized). A passage 'covers' a
    gold unit if it contains that sentence as a substring."""
    ev = q.get("evidence") or []
    sents = set()
    for e in ev:
        if isinstance(e, (list, tuple)) and len(e) >= 2 and isinstance(e[1], list):
            for s in e[1]:
                s = norm(s)
                if len(s) >= 12:  # skip trivially short fragments
                    sents.add(s)
    return sents


BASELINE_KW = dict(residual_strength=0.0, threshold_mode="absolute")
RESI_KW = dict(residual_strength=1.0, residual_k=3, threshold_mode="relative",
               delta_rel=0.4, adaptive=True, adapt_lo=2, adapt_hi=20)


def rrf(list_a, list_b, k=60, top=5):
    score = {}
    for lst in (list_a, list_b):
        for rank, p in enumerate(lst):
            score[p["id"]] = score.get(p["id"], 0.0) + 1.0 / (k + rank + 1)
    by_id = {p["id"]: p for p in list_a + list_b}
    ranked = sorted(score, key=lambda i: -score[i])[:top]
    return [by_id[i] for i in ranked]


def retrieve(retr, question, top_k, mode, esc_K=3, base_delta=None):
    """Return top_k passages for a condition.
      baseline    - original LinearRAG
      resirag     - always apply the residual/relative retriever (destructive)
      resirag-fuse- escalate only when baseline activation is starved (<=esc_K)
                    and fuse baseline+residual rankings with RRF (non-destructive)
    """
    bkw = dict(BASELINE_KW, delta=base_delta) if base_delta is not None else BASELINE_KW
    if mode == "baseline":
        return retr(question, top_k=top_k, **bkw)["passages"]
    if mode == "resirag":
        return retr(question, top_k=top_k, **RESI_KW)["passages"]
    # resirag-fuse
    rb = retr(question, top_k=max(top_k, 20), **BASELINE_KW)
    if len(rb["activated_entities"]) > esc_K:
        return rb["passages"][:top_k]
    re = retr(question, top_k=max(top_k, 20), **RESI_KW)
    return rrf(rb["passages"], re["passages"], top=top_k)


def covered_titles(passages: list[dict]) -> set[str]:
    return {p["title"] for p in passages if p.get("title")}


def covered_sentences(passages: list[dict], gold_sents: set[str]) -> set[str]:
    joined = " ".join(norm(p["text"]) for p in passages)
    return {s for s in gold_sents if s in joined}


def evaluate(index, retr, qs, strategy, top_k, mode, corpus_titles,
             base_delta=None):
    contain = title_recall = allgold = 0
    n_gold_q = 0
    for q in qs:
        ps = retrieve(retr, q["question"], top_k, mode, base_delta=base_delta)
        # Contain@k (answer OR any aliased surface form)
        answers = [norm(q["answer"])] + [norm(a) for a in q.get("answer_aliases", [])]
        blob = " ".join(norm(p["text"]) for p in ps)
        if any(a in blob for a in answers if a):
            contain += 1
        # gold recall
        if strategy == "ids":
            gold = set(q.get("gold_ids", []))
            got = {p["id"] for p in ps} & gold
        elif strategy == "title":
            gold = gold_units_title(q, corpus_titles)
            got = covered_titles(ps) & gold
        else:
            gold = gold_units_sentence(q)
            got = covered_sentences(ps, gold)
        if gold:
            n_gold_q += 1
            title_recall += len(got) / len(gold)
            if got == gold:
                allgold += 1
    n = len(qs)
    return {
        "Contain@k": contain / n,
        "GoldRecall": title_recall / n_gold_q if n_gold_q else float("nan"),
        "AllGoldHit@k": allgold / n_gold_q if n_gold_q else float("nan"),
        "n": n, "n_gold": n_gold_q,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--strategy", choices=["ids", "title", "sentence"],
                    required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--type", default=None)
    ap.add_argument("--skip-yesno", action="store_true")
    ap.add_argument("--modes", default="baseline,resirag-fuse",
                    help="comma list of baseline|resirag|resirag-fuse")
    ap.add_argument("--base-delta", type=float, default=None,
                    help="override baseline delta (calibration sweeps)")
    args = ap.parse_args()

    index = TriGraphIndex.load(args.index)
    retr = ResiRetriever(index, Embedder(index.meta["embedding_model"]),
                         load_nlp(index.meta["spacy_model"]))
    corpus_titles = ({json.loads(l)["title"]
                      for l in open(f"{args.index}/passages.jsonl",
                                    encoding="utf-8")}
                     if args.strategy == "title" else set())

    qs = json.load(open(args.questions, encoding="utf-8"))
    if args.type:
        qs = [q for q in qs if q.get("type") == args.type]
    if args.skip_yesno:
        qs = [q for q in qs if q["answer"] not in ("YES", "NO", "はい", "いいえ")]
    if args.limit:
        qs = qs[: args.limit]

    modes = args.modes.split(",")
    retr(qs[0]["question"], **BASELINE_KW)  # warm

    print(f"index={args.index}  n={len(qs)}  strategy={args.strategy}  "
          f"top_k={args.top_k}")
    print(f"{'condition':16s} {'Contain@k':>10s} {'GoldRecall':>11s} "
          f"{'AllGoldHit':>11s}")
    for mode in modes:
        m = evaluate(index, retr, qs, args.strategy, args.top_k, mode,
                     corpus_titles, base_delta=args.base_delta)
        print(f"{mode:16s} {m['Contain@k']:>10.1%} {m['GoldRecall']:>11.1%} "
              f"{m['AllGoldHit@k']:>11.1%}   (n_gold={m['n_gold']})")


if __name__ == "__main__":
    main()

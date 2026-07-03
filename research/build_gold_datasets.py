"""Convert the canonical HippoRAG-format multi-hop eval sets into the project's
corpus.jsonl + questions.json, PRESERVING per-question gold passage identity.

Source files (fetched from OSU-NLP-Group/HippoRAG reproduce/dataset):
  <name>.json         - questions with context/supporting_facts (2wiki,hotpot)
                        or paragraphs+is_supporting (musique)
  <name>_corpus.json  - the retrieval pool as [{title,text}], one para per title
                        for 2wiki/hotpot; musique has multiple paras per title.

Output dataset/<name>_hpr/:
  corpus.jsonl   {"id","title","text"}  (id = stable corpus index)
  questions.json {"id","question","answer","answer_aliases","type","gold_ids"}

Gold is stored as passage ids (not titles) so musique's same-title multi-para
hops are distinguished. This is the retrieval pool the multi-hop RAG literature
(HippoRAG2/PropRAG/BridgeRAG) reports on, so gold-passage recall is comparable.
"""
import json
import os
import sys

SRC = "/tmp"
OUT = "dataset"


def norm(s):
    return " ".join(s.split()).casefold()


def build(name, src_q, src_c):
    corpus = json.load(open(f"{SRC}/{src_c}", encoding="utf-8"))
    qs = json.load(open(f"{SRC}/{src_q}", encoding="utf-8"))
    outdir = f"{OUT}/{name}_hpr"
    os.makedirs(outdir, exist_ok=True)

    # write corpus + build lookups
    title2id = {}
    normtext2id = {}
    with open(f"{outdir}/corpus.jsonl", "w", encoding="utf-8") as f:
        for i, p in enumerate(corpus):
            pid = str(i)
            title, text = p["title"], p["text"]
            f.write(json.dumps({"id": pid, "title": title, "text": text},
                               ensure_ascii=False) + "\n")
            title2id.setdefault(title, pid)          # first para of a title
            normtext2id[(title, norm(text))] = pid

    out_q = []
    miss = 0
    for q in qs:
        if name == "musique":
            sup = [p for p in q["paragraphs"] if p.get("is_supporting")]
            gold = []
            for p in sup:
                key = (p["title"], norm(p["paragraph_text"]))
                pid = normtext2id.get(key) or title2id.get(p["title"])
                if pid is None:
                    miss += 1
                else:
                    gold.append(pid)
            qid = q["id"]
            qtype = qid.split("__")[0].replace("musique_", "")  # 2hop/3hop/4hop
            rec = {"id": qid, "question": q["question"], "answer": q["answer"],
                   "answer_aliases": q.get("answer_aliases", []),
                   "type": qtype, "gold_ids": sorted(set(gold), key=int)}
        else:  # 2wiki, hotpot: supporting_facts = [[title, sent_id], ...]
            gold_titles = []
            for t, _ in q.get("supporting_facts", []):
                if t not in gold_titles:
                    gold_titles.append(t)
            gold = [title2id[t] for t in gold_titles if t in title2id]
            miss += sum(1 for t in gold_titles if t not in title2id)
            rec = {"id": q.get("_id") or q.get("id"), "question": q["question"],
                   "answer": q["answer"], "answer_aliases": [],
                   "type": q.get("type", ""), "gold_ids": gold}
        out_q.append(rec)

    json.dump(out_q, open(f"{outdir}/questions.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    ng = [len(r["gold_ids"]) for r in out_q]
    print(f"{name:8s} corpus={len(corpus):6d} q={len(out_q):5d}  "
          f"gold/q min={min(ng)} max={max(ng)} "
          f"mean={sum(ng)/len(ng):.2f}  unmatched_gold={miss}  -> {outdir}")


if __name__ == "__main__":
    build("2wiki", "2wiki_q.json", "2wiki_c.json")
    build("hotpot", "hotpot_q.json", "hotpot_c.json")
    build("musique", "musique_q.json", "musique_c.json")

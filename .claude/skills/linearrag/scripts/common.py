"""LinearRAG Tri-Graph index: shared data structures and I/O.

Implements the indexing layer of LinearRAG (arXiv:2510.10114) directly from
the paper's equations. Written independently of the reference implementation.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

_JA_CHARS = re.compile(r"[぀-ヿ一-鿿]")

SPACY_MODELS = {"en": "en_core_web_sm", "ja": "ja_core_news_sm"}
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"


def detect_language(texts: list[str], sample_size: int = 50) -> str:
    """Return "ja" when Japanese characters dominate a sample of the corpus."""
    sample = "".join(texts[:sample_size])
    if not sample:
        return "en"
    ratio = len(_JA_CHARS.findall(sample)) / len(sample)
    return "ja" if ratio > 0.1 else "en"


def normalize_entity(surface: str) -> str:
    """NFKC-normalize, casefold, and collapse whitespace so mentions align."""
    return " ".join(unicodedata.normalize("NFKC", surface).casefold().split())


@dataclass
class TriGraphIndex:
    """Relation-free Tri-Graph: passage/sentence/entity nodes, C and M edges."""

    passages: list[dict]        # {"id", "title", "text"}
    sentences: list[dict]       # {"text", "passage": passage row index}
    entities: list[str]         # normalized surface forms; row index = entity id
    C: sparse.csr_matrix        # |P| x |E| occurrence counts (paper eq.1; counts kept for eq.7)
    M: sparse.csr_matrix        # |S| x |E| binary mentions (paper eq.2)
    emb_passages: np.ndarray
    emb_sentences: np.ndarray
    emb_entities: np.ndarray
    meta: dict

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "passages.jsonl", "w", encoding="utf-8") as f:
            for p in self.passages:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        with open(out / "sentences.jsonl", "w", encoding="utf-8") as f:
            for s in self.sentences:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        (out / "entities.json").write_text(
            json.dumps(self.entities, ensure_ascii=False), encoding="utf-8")
        sparse.save_npz(out / "C.npz", self.C)
        sparse.save_npz(out / "M.npz", self.M)
        np.save(out / "emb_passages.npy", self.emb_passages)
        np.save(out / "emb_sentences.npy", self.emb_sentences)
        np.save(out / "emb_entities.npy", self.emb_entities)
        (out / "meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, in_dir: str | Path) -> "TriGraphIndex":
        d = Path(in_dir)
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"index not found: {d}")
        with open(d / "passages.jsonl", encoding="utf-8") as f:
            passages = [json.loads(line) for line in f]
        with open(d / "sentences.jsonl", encoding="utf-8") as f:
            sentences = [json.loads(line) for line in f]
        return cls(
            passages=passages,
            sentences=sentences,
            entities=json.loads((d / "entities.json").read_text(encoding="utf-8")),
            C=sparse.load_npz(d / "C.npz").tocsr(),
            M=sparse.load_npz(d / "M.npz").tocsr(),
            emb_passages=np.load(d / "emb_passages.npy"),
            emb_sentences=np.load(d / "emb_sentences.npy"),
            emb_entities=np.load(d / "emb_entities.npy"),
            meta=json.loads((d / "meta.json").read_text(encoding="utf-8")),
        )

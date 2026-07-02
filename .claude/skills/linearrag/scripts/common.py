"""LinearRAG Tri-Graph index: shared data structures and I/O.

Implements the indexing layer of LinearRAG (arXiv:2510.10114) directly from
the paper's equations. Written independently of the reference implementation.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

_JA_CHARS = re.compile(r"[぀-ヿ一-鿿]")

SPACY_MODELS = {"en": "en_core_web_sm", "ja": "ja_core_news_sm"}
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

#: On-disk layout version. Bump when the set/shape of index files changes.
INDEX_FORMAT = 1


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


def make_analyzer(nlp):
    """Return analyze(text) -> [(sentence, [entity surface, ...]), ...].

    Sentence segmentation and NER are driven by a single spaCy pass so each
    sentence carries exactly the entities detected within it. Blank sentences
    are dropped.
    """
    def analyze(text: str):
        doc = nlp(text)
        return [(sent.text.strip(), [ent.text for ent in sent.ents])
                for sent in doc.sents if sent.text.strip()]
    return analyze


@dataclass(eq=False)
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
        """Persist the index atomically.

        Files are written to a sibling temp directory and swapped into place
        with ``os.replace``, so an interrupted rebuild leaves either the old
        index intact or the new index complete -- never a half-written mix.
        A prior index is renamed aside during the swap and restored if the
        swap fails. ``out_dir`` must be on the same filesystem as its parent
        (``os.replace`` cannot cross filesystems).
        """
        out = Path(out_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.parent / f".tmp-{out.name}.{os.getpid()}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        try:
            with open(tmp / "passages.jsonl", "w", encoding="utf-8") as f:
                for p in self.passages:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            with open(tmp / "sentences.jsonl", "w", encoding="utf-8") as f:
                for s in self.sentences:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            (tmp / "entities.json").write_text(
                json.dumps(self.entities, ensure_ascii=False), encoding="utf-8")
            sparse.save_npz(tmp / "C.npz", self.C)
            sparse.save_npz(tmp / "M.npz", self.M)
            np.save(tmp / "emb_passages.npy", self.emb_passages)
            np.save(tmp / "emb_sentences.npy", self.emb_sentences)
            np.save(tmp / "emb_entities.npy", self.emb_entities)
            meta = {**self.meta, "index_format": INDEX_FORMAT}
            (tmp / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            # Swap into place. A prior index is renamed aside (not deleted)
            # so it can be restored if the final rename fails; os.replace
            # onto a non-empty directory is not portable across platforms.
            old_backup = out.parent / f".bak-{out.name}.{os.getpid()}"
            for stale in out.parent.glob(f".bak-{out.name}.*"):
                shutil.rmtree(stale)
            if out.exists():
                os.replace(out, old_backup)
            try:
                os.replace(tmp, out)
            except BaseException:
                if old_backup.exists():
                    os.replace(old_backup, out)  # restore the old index
                raise
            else:
                if old_backup.exists():
                    shutil.rmtree(old_backup)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)

    @classmethod
    def load(cls, in_dir: str | Path) -> "TriGraphIndex":
        d = Path(in_dir)
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"index not found: {d}")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        fmt = meta.get("index_format")
        if fmt != INDEX_FORMAT:
            raise ValueError(
                f"unsupported index_format {fmt!r} in {d / 'meta.json'} "
                f"(expected {INDEX_FORMAT})")
        with open(d / "passages.jsonl", encoding="utf-8") as f:
            passages = [json.loads(line) for line in f if line.strip()]
        with open(d / "sentences.jsonl", encoding="utf-8") as f:
            sentences = [json.loads(line) for line in f if line.strip()]
        entities = json.loads((d / "entities.json").read_text(encoding="utf-8"))
        index = cls(
            passages=passages,
            sentences=sentences,
            entities=entities,
            C=sparse.load_npz(d / "C.npz").tocsr(),
            M=sparse.load_npz(d / "M.npz").tocsr(),
            emb_passages=np.load(d / "emb_passages.npy"),
            emb_sentences=np.load(d / "emb_sentences.npy"),
            emb_entities=np.load(d / "emb_entities.npy"),
            meta=meta,
        )
        index._validate(d)
        return index

    def _validate(self, d: Path) -> None:
        """Check that matrix/embedding shapes agree with node counts."""
        n_p, n_s, n_e = len(self.passages), len(self.sentences), len(self.entities)
        checks = [
            (self.C.shape == (n_p, n_e), "C.npz",
             f"C.shape {self.C.shape} != (passages {n_p}, entities {n_e})"),
            (self.M.shape == (n_s, n_e), "M.npz",
             f"M.shape {self.M.shape} != (sentences {n_s}, entities {n_e})"),
            (self.emb_passages.shape[0] == n_p, "emb_passages.npy",
             f"emb_passages rows {self.emb_passages.shape[0]} != passages {n_p}"),
            (self.emb_sentences.shape[0] == n_s, "emb_sentences.npy",
             f"emb_sentences rows {self.emb_sentences.shape[0]} != sentences {n_s}"),
            (self.emb_entities.shape[0] == n_e, "emb_entities.npy",
             f"emb_entities rows {self.emb_entities.shape[0]} != entities {n_e}"),
        ]
        for ok, fname, msg in checks:
            if not ok:
                raise ValueError(f"corrupt index at {d / fname}: {msg}")

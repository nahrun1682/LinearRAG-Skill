# LinearRAG Skill Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** LinearRAG論文 (arXiv:2510.10114v4) のTri-Graph構築と2段階グラフ検索を、単一のClaude Codeスキル `.claude/skills/linearrag/` としてPythonで実装する。

**Architecture:** 論文の数式（式1〜7）を scipy 疎行列でベクトル化実装する。`common.py` に索引データ構造とI/O、`build_graph.py` にトークンフリーなグラフ構築（文分割→spaCy NER→C/M行列→埋め込み）、`query_graph.py` に2段階検索（式5の意味伝播＋動的枝刈り → 式7ハイブリッド初期化＋Personalized PageRank）を置く。回答生成はPythonでは行わず、top-kパッセージをJSONで返してClaudeが生成する。

**Tech Stack:** Python 3.11+, uv (PEP 723 inline metadata), numpy, scipy, spaCy (en_core_web_sm / ja_core_news_sm), sentence-transformers, pytest。

**設計書:** `docs/plans/2026-07-02-linearrag-skill-design.md`

---

## ⚠️ ライセンス上の制約（全タスク共通・厳守）

公式実装 https://github.com/DEEP-PolyU/LinearRAG のコードを**コピー・翻案してはならない**。
実装は論文の数式から独自に行う。公式リポジトリから取り込んでよいのは以下の**事実情報のみ**:

| 項目 | 公式実装で確認した事実 | 本実装での扱い |
|---|---|---|
| δ (枝刈り閾値) | `iteration_threshold = 0.5` | デフォルト `--delta 0.5`（論文本文の「δ=4」は誤記と判断） |
| 最大反復 | `max_iterations = 3` | デフォルト4（論文の n≤4 に従う）、`--max-iterations` で変更可 |
| PPR damping | `0.5` | デフォルト `--damping 0.5`（論文本文の0.85は「typically」の一般論） |
| 式7の λ | DPRスコアを min-max 正規化後 1.5 倍 | `lam = 1.5` |
| 式7の W_p | `passage_node_weight = 0.05` | `w_p = 0.05` |
| 式7の L_ei | エンティティが活性化された反復番号（シード=1、以降+1） | 同じ意味論を採用 |

**構造上の意図的な差異**（コピー回避のためではなく設計として優れているため）:
- 公式: igraph + PRPACK、BFSループ型伝播、passage-passage エッジあり、ハッシュIDストア
- 本実装: 純粋な scipy 疎行列演算（式5を文字通りベクトル化）、自前べき乗法PPR、
  論文通りの二部グラフのみ、jsonl/npz の平易な索引フォーマット、日英対応

---

## Task 1: 環境確認とテスト基盤

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/conftest.py`

**Step 1: uv の存在確認**

Run: `uv --version`
Expected: バージョン表示。エラーなら `winget install astral-sh.uv`（または WSL 側で `curl -LsSf https://astral.sh/uv/install.sh | sh`）を人間のパートナーに提案して停止。

**Step 2: pyproject.toml に dev 依存を追加**

ユニットテストは numpy/scipy のみで動く設計（spaCy・sentence-transformers は遅延import）なので、devグループは最小限:

```toml
[project]
name = "linearrag-skill"
version = "0.1.0"
description = "LinearRAG (arXiv:2510.10114) as a Claude Code skill"
readme = "README.md"
requires-python = ">=3.13"
dependencies = []

[dependency-groups]
dev = [
    "pytest>=8.0",
    "numpy>=2.0",
    "scipy>=1.14",
]
```

**Step 3: conftest.py でスキルの scripts/ を import パスに追加**

```python
# tests/conftest.py
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.parent / ".claude" / "skills" / "linearrag" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
```

**Step 4: pytest が起動することを確認**

Run: `uv run pytest tests/ -v`
Expected: `no tests ran`（エラーなく終了すること）

**Step 5: Commit**

```bash
git add pyproject.toml tests/conftest.py uv.lock
git commit -m "chore: add dev dependencies and test scaffolding"
```

---

## Task 2: common.py — 言語判定とエンティティ正規化

**Files:**
- Create: `.claude/skills/linearrag/scripts/common.py`
- Test: `tests/test_common.py`

**Step 1: 失敗するテストを書く**

```python
# tests/test_common.py
from common import detect_language, normalize_entity


def test_detect_language_english():
    assert detect_language(["The quick brown fox jumps over the lazy dog."]) == "en"


def test_detect_language_japanese():
    assert detect_language(["東京は日本の首都です。", "大阪は西日本の中心都市です。"]) == "ja"


def test_detect_language_empty():
    assert detect_language([]) == "en"


def test_normalize_entity_casefold_and_whitespace():
    assert normalize_entity("  Frederick   Barbarossa ") == "frederick barbarossa"
    assert normalize_entity("TOKYO") == "tokyo"
```

**Step 2: 失敗を確認**

Run: `uv run pytest tests/test_common.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'common'`)

**Step 3: 最小実装**

```python
# .claude/skills/linearrag/scripts/common.py
"""LinearRAG Tri-Graph index: shared data structures and I/O.

Implements the indexing layer of LinearRAG (arXiv:2510.10114) directly from
the paper's equations. Written independently of the reference implementation.
"""
from __future__ import annotations

import json
import re
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
    """Casefold and collapse whitespace so mentions align across passages."""
    return " ".join(surface.casefold().split())
```

**Step 4: パスを確認**

Run: `uv run pytest tests/test_common.py -v`
Expected: 4 PASS

**Step 5: Commit**

```bash
git add .claude/skills/linearrag/scripts/common.py tests/test_common.py
git commit -m "feat: add language detection and entity normalization"
```

---

## Task 3: TriGraphIndex の保存・読み込み

**Files:**
- Modify: `.claude/skills/linearrag/scripts/common.py`
- Test: `tests/test_index_io.py`

**Step 1: 失敗するテストを書く**

```python
# tests/test_index_io.py
import numpy as np
from scipy import sparse

from common import TriGraphIndex


def _tiny_index() -> TriGraphIndex:
    return TriGraphIndex(
        passages=[{"id": "p0", "title": "", "text": "alpha beta."},
                  {"id": "p1", "title": "T", "text": "beta gamma."}],
        sentences=[{"text": "alpha beta.", "passage": 0},
                   {"text": "beta gamma.", "passage": 1}],
        entities=["alpha", "beta", "gamma"],
        C=sparse.csr_matrix(np.array([[1, 1, 0], [0, 1, 2]])),
        M=sparse.csr_matrix(np.array([[1, 1, 0], [0, 1, 1]])),
        emb_passages=np.eye(2, 4, dtype=np.float32),
        emb_sentences=np.eye(2, 4, dtype=np.float32),
        emb_entities=np.eye(3, 4, dtype=np.float32),
        meta={"language": "en", "embedding_model": "dummy"},
    )


def test_save_load_roundtrip(tmp_path):
    index = _tiny_index()
    index.save(tmp_path / "idx")
    loaded = TriGraphIndex.load(tmp_path / "idx")

    assert loaded.passages == index.passages
    assert loaded.sentences == index.sentences
    assert loaded.entities == index.entities
    assert (loaded.C != index.C).nnz == 0
    assert (loaded.M != index.M).nnz == 0
    np.testing.assert_array_equal(loaded.emb_entities, index.emb_entities)
    assert loaded.meta["language"] == "en"


def test_load_missing_dir_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        TriGraphIndex.load(tmp_path / "nope")
```

**Step 2: 失敗を確認**

Run: `uv run pytest tests/test_index_io.py -v`
Expected: FAIL (`ImportError: cannot import name 'TriGraphIndex'`)

**Step 3: 実装（common.py に追記）**

```python
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
```

**Step 4: パスを確認**

Run: `uv run pytest tests/test_index_io.py -v`
Expected: 2 PASS

**Step 5: Commit**

```bash
git add .claude/skills/linearrag/scripts/common.py tests/test_index_io.py
git commit -m "feat: add TriGraphIndex with save/load"
```

---

## Task 4: コーパス読み込み（3形式）

**Files:**
- Create: `.claude/skills/linearrag/scripts/build_graph.py`
- Test: `tests/test_corpus_loading.py`

**Step 1: 失敗するテストを書く**

```python
# tests/test_corpus_loading.py
import json

from build_graph import load_corpus


def test_load_dataset_chunks_json(tmp_path):
    # dataset/ 形式: "id:本文" の文字列リスト
    chunks = ["0:first passage text.", "1:second passage: with colon."]
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps(chunks), encoding="utf-8")

    passages = load_corpus(path)
    assert passages == [
        {"id": "0", "title": "", "text": "first passage text."},
        {"id": "1", "title": "", "text": "second passage: with colon."},
    ]


def test_load_jsonl(tmp_path):
    path = tmp_path / "corpus.jsonl"
    rows = [{"id": "a", "title": "T1", "text": "hello."},
            {"id": "b", "text": "world."}]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    passages = load_corpus(path)
    assert passages[0] == {"id": "a", "title": "T1", "text": "hello."}
    assert passages[1] == {"id": "b", "title": "", "text": "world."}


def test_load_directory_of_text_files(tmp_path):
    (tmp_path / "doc1.txt").write_text("Para one.\n\nPara two.", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "doc2.md").write_text("# Heading\n\nBody text.", encoding="utf-8")

    passages = load_corpus(tmp_path)
    ids = [p["id"] for p in passages]
    assert "doc1.txt#0" in ids
    assert any(i.startswith("sub/doc2.md" if "/" in i else "sub\\doc2.md") for i in ids)
    assert all(p["text"] for p in passages)


def test_directory_merges_small_paragraphs(tmp_path):
    paras = "\n\n".join(f"Short para {i}." for i in range(10))
    (tmp_path / "doc.txt").write_text(paras, encoding="utf-8")
    passages = load_corpus(tmp_path, max_chars=200)
    # 10段落が max_chars を上限に少数のパッセージへ統合される
    assert 1 <= len(passages) < 10
    assert all(len(p["text"]) <= 200 + 50 for p in passages)
```

**Step 2: 失敗を確認**

Run: `uv run pytest tests/test_corpus_loading.py -v`
Expected: FAIL (`No module named 'build_graph'`)

**Step 3: 実装**

`build_graph.py` の先頭にはPEP 723ブロックを置く（Task 7で実行時に使用）:

```python
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy>=2.0",
#     "scipy>=1.14",
#     "spacy>=3.8",
#     "sentence-transformers>=3.0",
#     "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#     "ja-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/ja_core_news_sm-3.8.0/ja_core_news_sm-3.8.0-py3-none-any.whl",
# ]
# ///
"""Build a LinearRAG Tri-Graph index from a corpus (paper §3.1, token-free)."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

from common import (DEFAULT_EMBEDDING_MODEL, SPACY_MODELS, TriGraphIndex,
                    detect_language, normalize_entity)


def _split_paragraphs(text: str, max_chars: int) -> list[str]:
    """Merge blank-line-separated paragraphs up to max_chars per passage."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    merged: list[str] = []
    buf = ""
    for para in paras:
        if buf and len(buf) + len(para) + 1 > max_chars:
            merged.append(buf)
            buf = para
        else:
            buf = f"{buf}\n{para}" if buf else para
    if buf:
        merged.append(buf)
    return merged


def load_corpus(path: str | Path, max_chars: int = 1000) -> list[dict]:
    """Load passages from a directory (.txt/.md), a .jsonl file, or a
    dataset-style chunks.json ("id:text" string list)."""
    path = Path(path)
    passages: list[dict] = []

    if path.is_dir():
        for file in sorted(path.rglob("*")):
            if file.suffix.lower() not in {".txt", ".md"}:
                continue
            rel = file.relative_to(path).as_posix()
            text = file.read_text(encoding="utf-8")
            for i, chunk in enumerate(_split_paragraphs(text, max_chars)):
                passages.append({"id": f"{rel}#{i}", "title": rel, "text": chunk})
        return passages

    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                passages.append({"id": str(row["id"]), "title": row.get("title", ""),
                                 "text": row["text"]})
        return passages

    # dataset-style chunks.json: list of "id:text" strings
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data:
        pid, _, text = item.partition(":")
        passages.append({"id": pid, "title": "", "text": text})
    return passages
```

**Step 4: パスを確認**

Run: `uv run pytest tests/test_corpus_loading.py -v`
Expected: 4 PASS

**Step 5: Commit**

```bash
git add .claude/skills/linearrag/scripts/build_graph.py tests/test_corpus_loading.py
git commit -m "feat: add corpus loading for directory, jsonl, and chunks.json"
```

---

## Task 5: Tri-Graph 組み立て（C・M 行列）

**Files:**
- Modify: `.claude/skills/linearrag/scripts/build_graph.py`
- Test: `tests/test_graph_assembly.py`

設計ポイント: 文分割・NER・埋め込みは**注入可能な callable** にする。ユニットテストは
モデル不要のフェイクで検証し、本物のspaCy/sentence-transformersはCLI層（Task 6）で接続する。

**Step 1: 失敗するテストを書く**

```python
# tests/test_graph_assembly.py
import numpy as np

from build_graph import assemble_tri_graph

PASSAGES = [
    {"id": "p0", "title": "", "text": "Alice met Bob. Bob lives in Tokyo."},
    {"id": "p1", "title": "", "text": "Tokyo is big. Alice visited Tokyo."},
]

# フェイク: ". " で文分割、大文字始まりの語をエンティティとみなす
def fake_split(text):
    return [s.strip() + "." for s in text.rstrip(".").split(". ")]

def fake_ner(sentence):
    return [w.strip(".") for w in sentence.split() if w[0].isupper()]

def fake_embed(texts):
    rng = np.random.default_rng(0)
    return rng.normal(size=(len(texts), 8)).astype(np.float32)


def test_assemble_builds_nodes_and_edges():
    index = assemble_tri_graph(PASSAGES, fake_split, fake_ner, fake_embed)

    assert len(index.passages) == 2
    assert len(index.sentences) == 4
    assert set(index.entities) == {"alice", "bob", "tokyo"}

    e = {name: i for i, name in enumerate(index.entities)}
    # C: 出現回数 (p1 に tokyo は2回)
    assert index.C[1, e["tokyo"]] == 2
    assert index.C[0, e["bob"]] == 2
    assert index.C[0, e["tokyo"]] == 1
    # M: 二値 (文0 = "Alice met Bob.")
    assert index.M[0, e["alice"]] == 1
    assert index.M[0, e["tokyo"]] == 0
    assert index.M.max() == 1

    # 文とパッセージの対応
    assert index.sentences[2]["passage"] == 1
    # 埋め込みの行数
    assert index.emb_passages.shape[0] == 2
    assert index.emb_sentences.shape[0] == 4
    assert index.emb_entities.shape[0] == 3


def test_assemble_skips_entityless_passages_gracefully():
    passages = [{"id": "x", "title": "", "text": "nothing here at all."}]
    index = assemble_tri_graph(passages, fake_split, fake_ner, fake_embed)
    assert index.C.shape == (1, 0)
    assert index.entities == []
```

**Step 2: 失敗を確認**

Run: `uv run pytest tests/test_graph_assembly.py -v`
Expected: FAIL (`cannot import name 'assemble_tri_graph'`)

**Step 3: 実装（build_graph.py に追記）**

```python
def assemble_tri_graph(passages, split_sentences, extract_entities, embed,
                       meta: dict | None = None) -> TriGraphIndex:
    """Build the Tri-Graph (paper §3.1): sentence/entity nodes plus the
    contain matrix C (counts) and mention matrix M (binary)."""
    sentences: list[dict] = []
    entity_ids: dict[str, int] = {}
    m_rows, m_cols = [], []
    c_rows, c_cols, c_vals = [], [], []

    for p_idx, passage in enumerate(passages):
        occurrence = Counter()
        for sent in split_sentences(passage["text"]):
            s_idx = len(sentences)
            sentences.append({"text": sent, "passage": p_idx})
            seen_in_sentence = set()
            for surface in extract_entities(sent):
                name = normalize_entity(surface)
                if not name:
                    continue
                e_idx = entity_ids.setdefault(name, len(entity_ids))
                occurrence[e_idx] += 1
                if e_idx not in seen_in_sentence:
                    seen_in_sentence.add(e_idx)
                    m_rows.append(s_idx)
                    m_cols.append(e_idx)
        for e_idx, count in occurrence.items():
            c_rows.append(p_idx)
            c_cols.append(e_idx)
            c_vals.append(count)

    entities = list(entity_ids)
    n_p, n_s, n_e = len(passages), len(sentences), len(entities)
    M = sparse.csr_matrix(
        (np.ones(len(m_rows), dtype=np.float32), (m_rows, m_cols)), shape=(n_s, n_e))
    C = sparse.csr_matrix(
        (np.asarray(c_vals, dtype=np.float32), (c_rows, c_cols)), shape=(n_p, n_e))

    return TriGraphIndex(
        passages=passages,
        sentences=sentences,
        entities=entities,
        C=C,
        M=M,
        emb_passages=embed([p["text"] for p in passages]),
        emb_sentences=embed([s["text"] for s in sentences]),
        emb_entities=embed(entities) if entities else np.zeros((0, 1), dtype=np.float32),
        meta=meta or {},
    )
```

**Step 4: パスを確認**

Run: `uv run pytest tests/test_graph_assembly.py -v`
Expected: 2 PASS

**Step 5: Commit**

```bash
git add .claude/skills/linearrag/scripts/build_graph.py tests/test_graph_assembly.py
git commit -m "feat: assemble Tri-Graph with contain/mention sparse matrices"
```

---

## Task 6: 実モデルアダプタ（spaCy・埋め込み）とビルドCLI

**Files:**
- Modify: `.claude/skills/linearrag/scripts/common.py`（Embedder, load_nlp を追加）
- Modify: `.claude/skills/linearrag/scripts/build_graph.py`（main を追加）

このタスクはモデルのダウンロードを伴うためユニットテストではなく、Task 7 の
スモークテストで検証する。**遅延importを厳守**（common.py のトップレベルで
spacy / sentence_transformers を import しない — ユニットテストが壊れる）。

**Step 1: common.py にアダプタを追加**

```python
class Embedder:
    """Lazy wrapper around SentenceTransformer with normalized outputs."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def __call__(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        vecs = self._model.encode(
            texts, batch_size=64, normalize_embeddings=True,
            show_progress_bar=len(texts) > 256)
        return np.asarray(vecs, dtype=np.float32)


def load_nlp(lang: str):
    """Load the spaCy pipeline for sentence splitting and NER."""
    import spacy
    return spacy.load(SPACY_MODELS[lang])


def make_splitter_and_ner(nlp):
    """Return (split_sentences, extract_entities) callables backed by spaCy.

    Sentences are cached per text so each passage is parsed once even though
    the two callables are invoked separately.
    """
    cache: dict[str, list] = {}

    def _doc_sents(text: str):
        if text not in cache:
            cache.clear()          # keep memory bounded: one passage at a time
            cache[text] = list(nlp(text).sents)
        return cache[text]

    def split_sentences(text: str) -> list[str]:
        return [s.text.strip() for s in _doc_sents(text) if s.text.strip()]

    def extract_entities(sentence: str) -> list[str]:
        for sents in cache.values():
            for s in sents:
                if s.text.strip() == sentence:
                    return [ent.text for ent in s.ents]
        return [ent.text for ent in nlp(sentence).ents]  # fallback

    return split_sentences, extract_entities
```

**注意:** 上記 `make_splitter_and_ner` のキャッシュ方式が複雑になるなら、
`assemble_tri_graph` を「パッセージ→(文, エンティティ列)リスト」を返す単一の
`analyze(text) -> list[tuple[str, list[str]]]` を受け取る形にリファクタしてよい
（テストのフェイクも同型に更新）。実装時にシンプルな方を選ぶこと。

**Step 2: build_graph.py に main を追加**

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Build a LinearRAG Tri-Graph index")
    parser.add_argument("--input", required=True, help="corpus dir / .jsonl / chunks.json")
    parser.add_argument("--output", required=True, help="index output directory")
    parser.add_argument("--lang", choices=["auto", "en", "ja"], default="auto")
    parser.add_argument("--model", default=None, help="sentence-transformers model name")
    parser.add_argument("--max-chars", type=int, default=1000)
    args = parser.parse_args()

    from common import Embedder, load_nlp, make_splitter_and_ner

    t0 = time.time()
    passages = load_corpus(args.input, max_chars=args.max_chars)
    if not passages:
        raise SystemExit(f"no passages found in {args.input}")

    lang = args.lang if args.lang != "auto" else detect_language(
        [p["text"] for p in passages])
    model_name = args.model or DEFAULT_EMBEDDING_MODEL
    print(f"passages={len(passages)} lang={lang} model={model_name}")

    nlp = load_nlp(lang)
    split_sentences, extract_entities = make_splitter_and_ner(nlp)
    embed = Embedder(model_name)

    meta = {"language": lang, "embedding_model": model_name,
            "spacy_model": SPACY_MODELS[lang], "num_passages": len(passages)}
    index = assemble_tri_graph(passages, split_sentences, extract_entities, embed, meta)
    index.save(args.output)

    print(f"sentences={len(index.sentences)} entities={len(index.entities)}")
    print(f"edges: C(nnz)={index.C.nnz} M(nnz)={index.M.nnz}")
    print(f"done in {time.time() - t0:.1f}s -> {args.output}")


if __name__ == "__main__":
    main()
```

**Step 3: ユニットテストが引き続き通ることを確認**

Run: `uv run pytest tests/ -v`
Expected: 全 PASS（遅延importにより numpy/scipy だけで動く）

**Step 4: Commit**

```bash
git add .claude/skills/linearrag/scripts/common.py .claude/skills/linearrag/scripts/build_graph.py
git commit -m "feat: wire spaCy/embedding adapters and build CLI"
```

---

## Task 7: ビルドのスモークテスト（実モデル）

**Files:** なし（手動検証タスク）

**Step 1: 極小サンプルで実行**

test10 の先頭50チャンクで試す（全658はTask 12で）:

```bash
cd <repo-root>
uv run python -c "
import json
chunks = json.load(open('dataset/test10/chunks.json', encoding='utf-8'))
json.dump(chunks[:50], open('/tmp/mini_chunks.json', 'w'))
"
uv run .claude/skills/linearrag/scripts/build_graph.py \
  --input /tmp/mini_chunks.json --output linearrag_index/mini --lang en
```

初回は uv が PEP 723 環境（torch含む、数GB）を構築するため時間がかかる。

Expected: `passages=50 lang=en`、entities > 100、エラーなく `linearrag_index/mini` に
8ファイル（passages.jsonl, sentences.jsonl, entities.json, C.npz, M.npz, emb_*.npy×3, meta.json）。

**Step 2: 索引を目視確認**

```bash
uv run python -c "
import sys; sys.path.insert(0, '.claude/skills/linearrag/scripts')
from common import TriGraphIndex
idx = TriGraphIndex.load('linearrag_index/mini')
print(idx.entities[:20]); print(idx.C.shape, idx.M.shape)
"
```

Expected: 正規化済みエンティティ名（人名・地名など）が並ぶこと。

**Step 3: .gitignore に索引出力を追加して Commit**

```bash
echo "linearrag_index/" >> .gitignore
git add .gitignore
git commit -m "chore: ignore built index outputs"
```

---

## Task 8: Stage 1 — エンティティ活性化（式3〜5＋動的枝刈り）

**Files:**
- Create: `.claude/skills/linearrag/scripts/query_graph.py`
- Test: `tests/test_activation.py`

**Step 1: 失敗するテストを書く**

2-hop の合成例: クエリエンティティ Beatrice → 文0 (Beatrice, Frederick) →
Frederick 活性化 → 文1 (Frederick, Germany) → Germany 活性化。

```python
# tests/test_activation.py
import numpy as np
from scipy import sparse

from query_graph import activate_entities, initial_activation

# entities: 0=beatrice, 1=frederick, 2=germany, 3=noise
# sentences: s0={beatrice,frederick}, s1={frederick,germany}, s2={noise}
M = sparse.csr_matrix(np.array([
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 1],
], dtype=np.float32))
SIGMA = np.array([0.9, 0.8, 0.1], dtype=np.float32)  # クエリ-文類似度
A0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # シード=beatrice


def test_two_hop_bridging_activates_intermediate_and_target():
    scores, levels, trace = activate_entities(A0, M, SIGMA, delta=0.5, max_iterations=4)
    assert scores[1] > 0.5          # frederick (1-hop bridge)
    assert scores[2] > 0.5          # germany (2-hop)
    assert scores[3] == 0.0         # noise は枝刈りされる
    assert levels[0] == 1           # シードは level 1
    assert levels[1] == 2
    assert levels[2] == 3
    assert levels[3] == 0           # 非活性


def test_pruning_threshold_blocks_weak_paths():
    scores, levels, _ = activate_entities(A0, M, SIGMA, delta=10.0, max_iterations=4)
    # 閾値が高すぎると新規活性はゼロ、シードのみ残る
    assert scores[0] == 1.0
    assert (scores[1:] == 0).all()


def test_terminates_early_when_no_new_entities():
    _, _, trace = activate_entities(A0, M, SIGMA, delta=0.5, max_iterations=10)
    assert len(trace) <= 3  # 3反復以内に自然停止（10回回らない）


def test_initial_activation_matches_most_similar_entity():
    emb_entities = np.array([[1, 0], [0, 1], [0.9, 0.1]], dtype=np.float32)
    emb_entities /= np.linalg.norm(emb_entities, axis=1, keepdims=True)
    q_ent_vecs = np.array([[1, 0]], dtype=np.float32)  # entity 0 に最も近い
    a0 = initial_activation(q_ent_vecs, emb_entities)
    assert a0.argmax() == 0
    assert a0[1] == 0.0
```

**Step 2: 失敗を確認**

Run: `uv run pytest tests/test_activation.py -v`
Expected: FAIL (`No module named 'query_graph'`)

**Step 3: 実装**

```python
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [  (build_graph.py と同一のブロックをコピー)
# ...
# ///
"""LinearRAG two-stage retrieval (paper §3.2): entity activation via semantic
bridging, then passage retrieval via personalized PageRank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from common import TriGraphIndex, normalize_entity


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
    Newly activated entities survive only if their score exceeds delta; the
    loop stops as soon as an iteration activates nothing new.

    Returns (scores, levels, trace): levels[i] is the 1-based iteration at
    which entity i first activated (0 = inactive) — used as L_ei in eq.7.
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
```

**Step 4: パスを確認**

Run: `uv run pytest tests/test_activation.py -v`
Expected: 4 PASS

（テスト前提の検算: spread₁ = Mᵀ(σ⊙(M·a₀)) → frederick = 0.9·1.0 = 0.9 ≥ 0.5 で活性。
spread₂ → germany = 0.8·0.9 = 0.72 ≥ 0.5 で活性。noise は文2経由のみで σ=0.1 → 0.1·0 = 0。）

**Step 5: Commit**

```bash
git add .claude/skills/linearrag/scripts/query_graph.py tests/test_activation.py
git commit -m "feat: stage-1 entity activation with semantic bridging and pruning"
```

---

## Task 9: Stage 2 前半 — パッセージ初期スコア（式7）

**Files:**
- Modify: `.claude/skills/linearrag/scripts/query_graph.py`
- Test: `tests/test_seed_scores.py`

**Step 1: 失敗するテストを書く**

```python
# tests/test_seed_scores.py
import numpy as np
from scipy import sparse

from query_graph import passage_seed_scores

# 2パッセージ × 3エンティティ。p0 は活性エンティティを多く含む。
C = sparse.csr_matrix(np.array([[3, 1, 0], [0, 0, 5]], dtype=np.float32))
A = np.array([0.9, 0.8, 0.0], dtype=np.float32)       # entity 2 は非活性
LEVELS = np.array([1, 2, 0], dtype=np.int32)
SIM_QP = np.array([0.5, 0.5], dtype=np.float32)       # DPR類似度は同点


def test_activated_entities_boost_containing_passage():
    scores = passage_seed_scores(SIM_QP, C, A, LEVELS, lam=1.5, w_p=0.05)
    assert scores[0] > scores[1]    # 活性エンティティを含む p0 が上
    assert (scores >= 0).all()


def test_deeper_entities_contribute_less():
    # 同じエンティティでも level が深いほど寄与が小さい (L_ei で除算)
    shallow = passage_seed_scores(SIM_QP, C, A, np.array([1, 1, 0]), lam=0.0, w_p=1.0)
    deep = passage_seed_scores(SIM_QP, C, A, np.array([3, 3, 0]), lam=0.0, w_p=1.0)
    assert shallow[0] > deep[0]


def test_no_activation_falls_back_to_dpr_only():
    scores = passage_seed_scores(np.array([0.9, 0.1]), C,
                                 np.zeros(3), np.zeros(3, dtype=np.int32),
                                 lam=1.5, w_p=0.05)
    assert scores[0] > scores[1]
```

**Step 2: 失敗を確認**

Run: `uv run pytest tests/test_seed_scores.py -v`
Expected: FAIL (`cannot import name 'passage_seed_scores'`)

**Step 3: 実装（query_graph.py に追記）**

```python
def _min_max(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def passage_seed_scores(sim_qp: np.ndarray, C: sparse.csr_matrix,
                        activation: np.ndarray, levels: np.ndarray,
                        lam: float = 1.5, w_p: float = 0.05) -> np.ndarray:
    """Paper eq.7: hybrid initialization of passage nodes.

    score(p) = (lam * minmax(sim(q,p)) + ln(1 + sum_i a_i * ln(1+N_pi) / L_i)) * w_p
    where N_pi is the occurrence count of entity i in passage p and L_i is the
    activation level from stage 1.
    """
    dpr = _min_max(sim_qp.astype(np.float64))
    C_log = C.astype(np.float64).copy()
    C_log.data = np.log1p(C_log.data)
    inv_level = np.divide(activation.astype(np.float64),
                          np.maximum(levels, 1),
                          out=np.zeros_like(activation, dtype=np.float64),
                          where=levels > 0)
    bonus = np.asarray(C_log @ inv_level).ravel()
    return ((lam * dpr + np.log1p(bonus)) * w_p).astype(np.float32)
```

**Step 4: パスを確認**

Run: `uv run pytest tests/test_seed_scores.py -v`
Expected: 3 PASS

**Step 5: Commit**

```bash
git add .claude/skills/linearrag/scripts/query_graph.py tests/test_seed_scores.py
git commit -m "feat: eq.7 hybrid passage seed scores"
```

---

## Task 10: Stage 2 後半 — Personalized PageRank（式6）

**Files:**
- Modify: `.claude/skills/linearrag/scripts/query_graph.py`
- Test: `tests/test_ppr.py`

**Step 1: 失敗するテストを書く**

```python
# tests/test_ppr.py
import numpy as np
from scipy import sparse

from query_graph import personalized_pagerank

# 3パッセージ × 2エンティティ: p0-{e0}, p1-{e0,e1}, p2-{e1}
B = sparse.csr_matrix(np.array([[1, 0], [1, 1], [0, 1]], dtype=np.float32))


def test_seeded_entity_lifts_connected_passages():
    passage_seeds = np.zeros(3, dtype=np.float32)
    entity_seeds = np.array([1.0, 0.0], dtype=np.float32)  # e0 のみシード
    p_scores, e_scores = personalized_pagerank(B, passage_seeds, entity_seeds,
                                               damping=0.5)
    assert p_scores[0] > p_scores[2]   # e0 に接続する p0 > 非接続の p2
    assert p_scores[1] > p_scores[2]


def test_scores_form_distribution():
    p, e = personalized_pagerank(B, np.ones(3, dtype=np.float32),
                                 np.ones(2, dtype=np.float32), damping=0.5)
    assert abs(p.sum() + e.sum() - 1.0) < 1e-6
    assert (p >= 0).all() and (e >= 0).all()


def test_uniform_fallback_when_no_seeds():
    p, e = personalized_pagerank(B, np.zeros(3, dtype=np.float32),
                                 np.zeros(2, dtype=np.float32), damping=0.5)
    assert abs(p.sum() + e.sum() - 1.0) < 1e-6
```

**Step 2: 失敗を確認**

Run: `uv run pytest tests/test_ppr.py -v`
Expected: FAIL (`cannot import name 'personalized_pagerank'`)

**Step 3: 実装（query_graph.py に追記）**

```python
def personalized_pagerank(B: sparse.csr_matrix, passage_seeds: np.ndarray,
                          entity_seeds: np.ndarray, damping: float = 0.5,
                          max_iter: int = 100, tol: float = 1e-9):
    """Paper eq.6: PPR over the passage-entity bipartite graph via power
    iteration. B is the binarized |P| x |E| biadjacency matrix. The reset
    distribution is the normalized concatenation of the two seed vectors.

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
```

**Step 4: パスを確認**

Run: `uv run pytest tests/test_ppr.py -v`
Expected: 3 PASS

**Step 5: 全テストのリグレッション確認**

Run: `uv run pytest tests/ -v`
Expected: 全 PASS

**Step 6: Commit**

```bash
git add .claude/skills/linearrag/scripts/query_graph.py tests/test_ppr.py
git commit -m "feat: personalized PageRank over passage-entity bipartite graph"
```

---

## Task 11: 検索CLIの結線と手動スモーク

**Files:**
- Modify: `.claude/skills/linearrag/scripts/query_graph.py`（retrieve + main）

**Step 1: retrieve と main を実装**

```python
def retrieve(index: TriGraphIndex, query: str, embed, nlp, top_k: int = 5,
             delta: float = 0.5, max_iterations: int = 4,
             lam: float = 1.5, w_p: float = 0.05, damping: float = 0.5) -> dict:
    query_vec = embed([query])[0]

    # --- Stage 1: entity activation (eq.3-5) ---
    q_entities = [normalize_entity(ent.text) for ent in nlp(query).ents]
    q_entities = [e for e in q_entities if e]
    q_vecs = embed(q_entities) if q_entities else np.zeros((0, 1), dtype=np.float32)
    a0 = initial_activation(q_vecs, index.emb_entities)
    sigma = index.emb_sentences @ query_vec
    activation, levels, trace = activate_entities(
        a0, index.M, sigma, delta=delta, max_iterations=max_iterations)

    # --- Stage 2: passage retrieval (eq.6-7) ---
    sim_qp = index.emb_passages @ query_vec
    p_seeds = passage_seed_scores(sim_qp, index.C, activation, levels,
                                  lam=lam, w_p=w_p)
    B = index.C.copy()
    B.data = np.ones_like(B.data)
    p_scores, _ = personalized_pagerank(B, p_seeds, activation, damping=damping)

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
                   "damping": damping, "max_iterations": max_iterations},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LinearRAG two-stage retrieval")
    parser.add_argument("--index", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--lam", type=float, default=1.5)
    parser.add_argument("--w-p", type=float, default=0.05)
    parser.add_argument("--damping", type=float, default=0.5)
    args = parser.parse_args()

    from common import Embedder, load_nlp

    index = TriGraphIndex.load(args.index)
    embed = Embedder(index.meta["embedding_model"])
    nlp = load_nlp(index.meta["language"])
    result = retrieve(index, args.query, embed, nlp, top_k=args.top_k,
                      delta=args.delta, max_iterations=args.max_iterations,
                      lam=args.lam, w_p=args.w_p, damping=args.damping)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

**Step 2: mini 索引でスモーク実行**

```bash
uv run .claude/skills/linearrag/scripts/query_graph.py \
  --index linearrag_index/mini --query "When did Lothair II's mother die?"
```

Expected: JSON が出力され、`activated_entities` に lothair 系のエンティティ、
`passages` に5件が score 降順で並ぶこと（mini索引に正解が含まれるかは問わない）。

**Step 3: 全テストのリグレッション確認**

Run: `uv run pytest tests/ -v`
Expected: 全 PASS

**Step 4: Commit**

```bash
git add .claude/skills/linearrag/scripts/query_graph.py
git commit -m "feat: wire two-stage retrieval CLI with JSON output"
```

---

## Task 12: test10 での結合評価

**Files:**
- Create: `.claude/skills/linearrag/scripts/evaluate.py`

**Step 1: フル索引を構築**

```bash
uv run .claude/skills/linearrag/scripts/build_graph.py \
  --input dataset/test10/chunks.json --output linearrag_index/test10
```

Expected: passages=658。CPUで数分〜十数分。

**Step 2: evaluate.py を実装**

```python
# /// script
# (build_graph.py と同一の PEP 723 ブロック)
# ///
"""Batch retrieval evaluation: Retrieval-Contain and Evidence-Entity Recall."""
from __future__ import annotations

import argparse
import json

import numpy as np

from common import Embedder, TriGraphIndex, load_nlp, normalize_entity
from query_graph import retrieve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--questions", required=True, help="questions.json path")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    index = TriGraphIndex.load(args.index)
    embed = Embedder(index.meta["embedding_model"])
    nlp = load_nlp(index.meta["language"])
    questions = json.loads(open(args.questions, encoding="utf-8").read())
    if args.limit:
        questions = questions[: args.limit]

    contain_hits, ev_hits, ev_total = 0, 0, 0
    for q in questions:
        result = retrieve(index, q["question"], embed, nlp, top_k=args.top_k)
        joined = " ".join(p["text"] for p in result["passages"]).casefold()
        contained = q["answer"].casefold() in joined
        contain_hits += contained

        activated = {r["entity"] for r in result["activated_entities"]}
        evidence_entities = {normalize_entity(step[0]) for step in q.get("evidence", [])}
        evidence_entities |= {normalize_entity(step[-1]) for step in q.get("evidence", [])}
        ev_total += len(evidence_entities)
        ev_hits += len(evidence_entities & activated)

        mark = "o" if contained else "x"
        print(f"[{mark}] {q['question'][:70]}  ans={q['answer'][:30]}")

    n = len(questions)
    print(f"\nRetrieval-Contain: {contain_hits}/{n} = {contain_hits / n:.1%}")
    if ev_total:
        print(f"Evidence-Entity Recall: {ev_hits}/{ev_total} = {ev_hits / ev_total:.1%}")


if __name__ == "__main__":
    main()
```

**Step 3: 評価を実行**

```bash
uv run .claude/skills/linearrag/scripts/evaluate.py \
  --index linearrag_index/test10 --questions dataset/test10/questions.json
```

Expected: **Retrieval-Contain ≥ 7/10**（設計書の成功基準）。

**Step 4: 未達の場合の調整手順（順に1つずつ変えて再計測）**

1. `--delta 0.3〜0.7` を振る
2. `--lam` を 1.0〜2.0、`--damping` を 0.5〜0.85 で振る
3. NERの取りこぼしを確認: 質問からエンティティが1つも取れないケースは
   `initial_activation` がゼロになり DPR フォールバックになる。en_core_web_sm →
   en_core_web_md への変更も検討（PEP 723 のURLを差し替え）
4. それでも未達なら結果と分析を人間のパートナーに報告して判断を仰ぐ

**Step 5: Commit**

```bash
git add .claude/skills/linearrag/scripts/evaluate.py
git commit -m "feat: add batch retrieval evaluation script"
```

---

## Task 13: SKILL.md と README

**Files:**
- Create: `.claude/skills/linearrag/SKILL.md`
- Modify: `README.md`

**Step 1: SKILL.md を書く**

```markdown
---
name: linearrag
description: Use when building a graph-RAG index from a document corpus or answering questions over an indexed corpus with LinearRAG (relation-free Tri-Graph retrieval, arXiv:2510.10114). Triggers include "コーパスに索引を張る", "グラフRAGで検索", "LinearRAGで質問に答える", "build a RAG index", "search the corpus".
---

# LinearRAG: Relation-Free Graph RAG

LLMトークン消費ゼロでTri-Graph（passage/sentence/entityノード＋C/M疎行列）を構築し、
2段階検索（意味伝播によるエンティティ活性化 → Personalized PageRank）で
multi-hop 質問に強いパッセージ検索を行う。回答生成はあなた（Claude）の仕事。

## ワークフロー1: 索引構築

対象コーパス（ディレクトリ / .jsonl / chunks.json）から索引を作る:

    uv run <このスキルのscripts>/build_graph.py \
      --input <コーパス> --output linearrag_index/<名前> [--lang auto|en|ja]

- 初回はモデルダウンロード（埋め込み約1GB + spaCy）で時間がかかると人間に伝えること
- 完了したら表示された統計（passages/sentences/entities/edges）を報告する

## ワークフロー2: 質問応答

    uv run <このスキルのscripts>/query_graph.py \
      --index linearrag_index/<名前> --query "<質問>" [--top-k 5]

出力JSONの読み方:
- `activated_entities`: 意味伝播が辿ったエンティティ連鎖（level=活性化した反復。
  multi-hop の推論チェーンとして回答の根拠説明に使える）
- `passages`: スコア順のtop-kパッセージ

**回答の生成手順:** passages の本文だけを根拠に回答する。根拠が不足していれば
その旨を述べる。activated_entities の連鎖を「どう辿って見つけたか」の説明に使う。

## ワークフロー3: 検索品質の評価（オプション）

questions.json（{question, answer, evidence}形式）があれば:

    uv run <このスキルのscripts>/evaluate.py \
      --index linearrag_index/<名前> --questions <questions.json>

## 調整パラメータ

| パラメータ | 既定値 | 効果 |
|---|---|---|
| --delta | 0.5 | 活性化の枝刈り閾値。下げると recall↑ noise↑ |
| --top-k | 5 | 返すパッセージ数 |
| --damping | 0.5 | PPRの減衰係数 |
| --lam / --w-p | 1.5 / 0.05 | 式7のDPR項とエンティティ項のバランス |
```

**Step 2: README.md を更新**

プロジェクト概要・論文リンク・スキルの使い方・テスト実行方法
（`uv run pytest tests/`）・ライセンス方針（独自実装であること）を簡潔に記載。

**Step 3: Commit**

```bash
git add .claude/skills/linearrag/SKILL.md README.md
git commit -m "docs: add SKILL.md and README"
```

---

## Task 14: E2E 検証と仕上げ

**Step 1: 全テスト最終確認**

Run: `uv run pytest tests/ -v`
Expected: 全 PASS

**Step 2: スキル経由のE2E**

スキルの SKILL.md の手順**だけ**に従って（このプランを見ずに）:
1. `linearrag_index/test10` に対して test10 の質問1問を query_graph.py で検索
2. 返ったパッセージから回答を生成
3. questions.json の正解と照合

Expected: 正答、または根拠パッセージがtop-5に含まれること。

**Step 3: 検証結果の記録とコミット**

@verification-before-completion に従い、実行した検証コマンドと出力を確認してから
完了を宣言する。

```bash
git add -A
git commit -m "test: verify end-to-end skill workflow on test10"
```

**Step 4: ブランチ完了処理**

@finishing-a-development-branch に従い、main へのマージ方法を人間のパートナーに確認。

import json

import numpy as np
import pytest
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
    with pytest.raises(FileNotFoundError):
        TriGraphIndex.load(tmp_path / "nope")


def test_dtype_preserved_on_roundtrip(tmp_path):
    index = _tiny_index()
    index.save(tmp_path / "idx")
    loaded = TriGraphIndex.load(tmp_path / "idx")

    assert loaded.C.dtype == index.C.dtype
    assert loaded.M.dtype == index.M.dtype
    assert loaded.emb_entities.dtype == np.float32
    assert loaded.emb_passages.dtype == np.float32
    assert loaded.emb_sentences.dtype == np.float32


def test_resave_over_existing_index_is_readable(tmp_path):
    out = tmp_path / "idx"
    _tiny_index().save(out)
    # Rebuild in place: a second save must leave a fully-readable index.
    _tiny_index().save(out)
    loaded = TriGraphIndex.load(out)
    assert loaded.entities == ["alpha", "beta", "gamma"]


def test_failed_resave_leaves_old_index_intact(tmp_path, monkeypatch):
    index = _tiny_index()
    index.save(tmp_path / "idx")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    # common.py calls np.save on the shared numpy module object, so patching
    # np.save here makes the second save fail mid-write, before the swap.
    monkeypatch.setattr(np, "save", boom)
    with pytest.raises(OSError):
        index.save(tmp_path / "idx")
    monkeypatch.undo()

    loaded = TriGraphIndex.load(tmp_path / "idx")  # old index still readable
    assert loaded.passages == index.passages
    assert not list(tmp_path.glob(".tmp-*"))       # no temp leftovers


def test_failed_swap_restores_old_index(tmp_path, monkeypatch):
    import os

    index = _tiny_index()
    index.save(tmp_path / "idx")

    real_replace = os.replace

    def flaky(src, dst):
        # Fail only the temp-dir -> index swap; let the backup renames
        # (index -> .bak-*, .bak-* -> index) go through.
        if ".tmp-" in str(src):
            raise OSError("swap failed")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    with pytest.raises(OSError, match="swap failed"):
        index.save(tmp_path / "idx")
    monkeypatch.undo()

    loaded = TriGraphIndex.load(tmp_path / "idx")  # old index restored
    assert loaded.passages == index.passages
    assert not list(tmp_path.glob(".tmp-*"))       # no temp leftovers
    assert not list(tmp_path.glob(".bak-*"))       # no backup leftovers


def test_save_leaves_no_temp_dir(tmp_path):
    out = tmp_path / "idx"
    _tiny_index().save(out)
    # Only the index dir should remain; no leftover temp/backup dirs beside it.
    siblings = {p.name for p in tmp_path.iterdir()}
    assert siblings == {"idx"}
    # And nothing temp-looking inside the index dir.
    assert all(not n.startswith(".tmp") for n in (p.name for p in out.iterdir()))


def test_save_injects_index_format(tmp_path):
    out = tmp_path / "idx"
    _tiny_index().save(out)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["index_format"] == 1


def test_load_rejects_unknown_index_format(tmp_path):
    out = tmp_path / "idx"
    _tiny_index().save(out)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    meta["index_format"] = 999
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError):
        TriGraphIndex.load(out)


def test_load_rejects_shape_mismatch(tmp_path):
    out = tmp_path / "idx"
    _tiny_index().save(out)
    # Tamper: add an entity so entities count no longer matches C/M columns.
    entities = json.loads((out / "entities.json").read_text(encoding="utf-8"))
    entities.append("delta")
    (out / "entities.json").write_text(
        json.dumps(entities, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="entities"):
        TriGraphIndex.load(out)

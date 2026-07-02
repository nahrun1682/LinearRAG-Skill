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

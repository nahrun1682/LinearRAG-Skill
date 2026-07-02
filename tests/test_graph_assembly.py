import numpy as np

from build_graph import assemble_tri_graph

PASSAGES = [
    {"id": "p0", "title": "", "text": "Alice met Bob. Bob lives in Tokyo."},
    {"id": "p1", "title": "", "text": "Tokyo is big. Alice visited Tokyo."},
]

# フェイク: ". " で文分割、大文字始まりの語をエンティティとみなす
def fake_analyze(text):
    sents = [s.strip() + "." for s in text.rstrip(".").split(". ") if s.strip()]
    return [(sent, [w.strip(".") for w in sent.split() if w[0].isupper()])
            for sent in sents]

def _make_fake_embed():
    rng = np.random.default_rng(0)
    def fake_embed(texts):
        return rng.normal(size=(len(texts), 8)).astype(np.float32)
    return fake_embed

fake_embed = _make_fake_embed()


def test_assemble_builds_nodes_and_edges():
    index = assemble_tri_graph(PASSAGES, fake_analyze, fake_embed)

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
    index = assemble_tri_graph(passages, fake_analyze, fake_embed)
    assert index.C.shape == (1, 0)
    assert index.M.shape == (1, 0)
    assert index.entities == []


def test_assemble_keeps_passages_with_no_sentences():
    # A passage whose splitter yields no sentences (e.g. empty text) still
    # occupies a passage row; C keeps a row for it and M gains no rows.
    passages = [
        {"id": "empty", "title": "", "text": ""},
        {"id": "p", "title": "", "text": "Alice met Bob."},
    ]
    index = assemble_tri_graph(passages, fake_analyze, fake_embed)
    assert len(index.passages) == 2
    # only the second passage contributes sentences
    assert len(index.sentences) == 1
    assert all(s["passage"] == 1 for s in index.sentences)
    assert index.C.shape[0] == 2
    assert index.M.shape[0] == 1

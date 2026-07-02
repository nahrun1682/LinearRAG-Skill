import json

import pytest

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
    assert any(i.startswith("sub/doc2.md") for i in ids)
    assert all(p["text"] for p in passages)


def test_directory_merges_small_paragraphs(tmp_path):
    paras = "\n\n".join(f"Short para {i}." for i in range(10))
    (tmp_path / "doc.txt").write_text(paras, encoding="utf-8")
    passages = load_corpus(tmp_path, max_chars=200)
    # 10段落が max_chars を上限に少数のパッセージへ統合される
    assert 1 <= len(passages) < 10
    # max_chars はマージ予算（ハード上限ではない）: 境界の1段落分の
    # はみ出しを許容するための +50 マージン。
    assert all(len(p["text"]) <= 200 + 50 for p in passages)


def test_directory_normalizes_crlf(tmp_path):
    # CRLF の空行 (\r\n\r\n) でも段落として分割されること
    (tmp_path / "doc.txt").write_bytes(b"Para one.\r\n\r\nPara two.")
    passages = load_corpus(tmp_path, max_chars=10)
    assert [p["text"] for p in passages] == ["Para one.", "Para two."]
    assert all("\r" not in p["text"] for p in passages)


def test_chunks_json_rejects_missing_colon(tmp_path):
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps(["0:ok", "no separator here"]), encoding="utf-8")
    with pytest.raises(ValueError, match="item 1"):
        load_corpus(path)


def test_chunks_json_rejects_non_string_item(tmp_path):
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps(["0:ok", {"id": 1}]), encoding="utf-8")
    with pytest.raises(TypeError, match="item 1.*dict"):
        load_corpus(path)

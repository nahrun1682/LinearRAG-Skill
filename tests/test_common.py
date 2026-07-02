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

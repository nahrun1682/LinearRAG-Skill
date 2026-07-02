# LinearRAG-Skill

論文 [LinearRAG: Linear Graph Retrieval-Augmented Generation on Large-Scale Corpora](https://arxiv.org/abs/2510.10114)
（arXiv:2510.10114）のグラフ構築と2段階グラフ検索を、Claude Code スキルとして Python で実装したもの。

- **Tri-Graph構築**（`build_graph.py`）: 文分割 + spaCy NER + 疎行列（contain行列C / mention行列M）
  + sentence-transformers 埋め込み。LLMトークン消費ゼロ・コーパスサイズに線形
- **2段階検索**（`query_graph.py`）: ①式5の意味伝播によるエンティティ活性化（動的枝刈り付き）
  ②式7のハイブリッド初期化 + Personalized PageRank によるパッセージ検索
- **回答生成はClaude**: スキルは top-k パッセージと活性化エンティティの推論チェーンをJSONで返し、
  呼び出し元のClaudeが回答を生成する

## 使い方

スキル本体: [.claude/skills/linearrag/SKILL.md](.claude/skills/linearrag/SKILL.md)

```bash
# 索引構築（例: dataset/test10）
uv run .claude/skills/linearrag/scripts/build_graph.py \
  --input dataset/test10/chunks.json --output linearrag_index/test10 \
  --spacy-model en_core_web_trf

# 検索
uv run .claude/skills/linearrag/scripts/query_graph.py \
  --index linearrag_index/test10 --query "When did Lothair II's mother die?"

# バッチ評価
uv run .claude/skills/linearrag/scripts/evaluate.py \
  --index linearrag_index/test10 --questions dataset/test10/questions.json
```

依存は各スクリプトの PEP 723 インラインメタデータで宣言されており、`uv run` が自動解決する。
日英両対応（言語自動判定、spaCy en/ja + 多言語埋め込みモデル）。

## テスト

```bash
uv run pytest tests/
```

ユニットテストは numpy/scipy のみで動く（spaCy・埋め込みモデルは遅延import＋注入可能なcallable設計）。

## 実装について

論文の数式（式1〜7）から独自に実装したクリーンルーム実装であり、
[公式実装](https://github.com/DEEP-PolyU/LinearRAG)のコードは使用していない
（公式からはハイパーパラメータの意味と値のみを事実情報として参照）。
設計の詳細は [docs/plans/](docs/plans/) の設計書・実装計画を参照。

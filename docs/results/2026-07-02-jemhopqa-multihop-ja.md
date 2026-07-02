# JEMHopQA dev 120問 — 日本語multi-hop検証結果

2026-07-02 ・ Task 19

## セットアップ

- **質問**: JEMHopQA ver1.2 dev（120問 = comparison 73 + compositional 47、うちYES/NO答45問、time_dependent 7問）
  https://github.com/aiishii/JEMHopQA （CC BY-SA 4.0）
- **コーパス**: derivationsのpage_ids全219記事をWikipedia API（2026年時点本文）から取得し、
  段落マージ（~1000字）で3,578パッセージ化。`dataset/jemhopqa/`（corpus.jsonl / questions.json）
- **索引**: GiNZA（ja_ginza 5.2.0、split_mode=C）+ paraphrase-multilingual-mpnet-base-v2。
  構築431秒、文83,182・エンティティ91,749（25.6個/パッセージ）
- **比較**: 同一埋め込み・同一コーパスで素のdense検索（cos類似 top-5）
- **指標**: Retrieval-Contain@5（正解文字列がtop-5パッセージに出現）

## 結果

| 区分 | n | LinearRAG | dense | 倍率 |
|---|---|---|---|---|
| 全体 | 120 | 38.3% | 25.8% | 1.5x |
| 実質評価可能（YES/NO除く） | 75 | **49.3%** | 30.7% | **1.6x** |
| 　comparison（実質） | 28 | **78.6%** | 46.4% | 1.7x |
| 　compositional（実質） | 47 | 31.9% | 21.3% | 1.5x |
| YES/NO答（指標対象外） | 45 | ~20% | ~18% | ノイズ |
| time_dependent | 7 | 1/7 | 0/7 | — |

Evidence-Entity Recall（LinearRAG）: 151/240 = **62.9%**

## 結論

- 英語（2Wiki 3.1x / HotpotQA 1.8x / MuSiQue 1.7x）で確認した「multi-hopでのグラフ優位」が
  **日本語でも1.5〜1.7xで再現**。GiNZA索引がグラフの背骨として機能した
- JSQuAD（単一ホップ、グラフが逆効果）と対照的に、multi-hopではGiNZA整備が効く
- compositionalの絶対値が低い主因は (a) 正解が記事深部にある（フル記事30パッセージ/記事）
  (b) 2021年正解 vs 2026年本文の時点ズレ（time_dependentはほぼ全滅）
- YES/NO 45問はContain指標では評価不能（生成評価の領域）

## 再現手順

```bash
# コーパス構築は /tmp/build_jem_corpus2.py 相当（Wikipedia API、リトライ付き）
uv run --with "ginza==5.2.0" --with "ja-ginza==5.2.0" \
  .claude/skills/linearrag/scripts/build_graph.py \
  --input dataset/jemhopqa/corpus.jsonl --output linearrag_index/jemhopqa \
  --lang ja --spacy-model ja_ginza
uv run --with "ginza==5.2.0" --with "ja-ginza==5.2.0" \
  .claude/skills/linearrag/scripts/evaluate.py \
  --index linearrag_index/jemhopqa --questions dataset/jemhopqa/questions.json
```

## 今後（Obsidian vault化に向けた含意）

- 日本語でも「連想が必要な検索」ならグラフが効くことが実証された
- vaultでは [[wikilink]]・タグをエンティティ昇格させればGiNZAへの依存をさらに下げられる
- 本格評価にはwikipedia-utils（2023/2024スナップショット）でのコーパス再構築とdistractor追加が候補

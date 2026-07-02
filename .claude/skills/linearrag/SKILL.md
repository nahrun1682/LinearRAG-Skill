---
name: linearrag
description: Use when building a graph-RAG index from a document corpus or answering questions over an indexed corpus with LinearRAG (relation-free Tri-Graph retrieval, arXiv:2510.10114). Triggers include "コーパスに索引を張る", "グラフRAGで検索", "LinearRAGで質問に答える", "build a RAG index", "search the corpus", "multi-hop QA over documents".
---

# LinearRAG: Relation-Free Graph RAG

LLMトークン消費ゼロでTri-Graph（passage / sentence / entity ノード＋contain行列C / mention行列M）を構築し、
2段階検索（①意味伝播によるエンティティ活性化 → ②Personalized PageRank）で multi-hop 質問に強い
パッセージ検索を行う。**回答生成はあなた（Claude）の仕事** — Pythonは検索までを担当する。

実行はすべて `uv run`（PEP 723 インラインメタデータで依存解決）。このリポジトリでは
**WSL内での実行が標準**: `wsl -e bash -lc 'cd /root/work/LinearRAG-Skill && uv run ...'`

## ワークフロー1: 索引構築

対象コーパス（ディレクトリの.txt/.md / .jsonl / dataset形式chunks.json）から索引を作る:

```bash
uv run .claude/skills/linearrag/scripts/build_graph.py \
  --input <コーパスパス> --output linearrag_index/<名前> \
  [--lang auto|en|ja] [--model <埋め込みモデル>] [--spacy-model <spaCyモデル>] \
  [--max-chars 1000]  # ディレクトリ入力時の段落マージ予算
```

- 初回はモデルダウンロード（埋め込み約1GB + spaCy）が走るため時間がかかると人間に伝えること
- **コーパスが小文字化済みの英語テキストの場合は `--spacy-model en_core_web_trf` を必ず指定**
  （デフォルトの en_core_web_sm は大文字手がかりに依存し、固有名詞をほぼ取りこぼす）
- **NVIDIA GPUがあるなら trf 使用時は `--gpu` を付け、`uv run --with "cupy-cuda12x<14"` で起動**:
  `uv run --with "cupy-cuda12x<14" .../build_graph.py ... --spacy-model en_core_web_trf --gpu`
  （実測: 658パッセージのtrf索引構築が 937.6s → 123.6s、7.6倍高速。索引内容は同一）
- 完了したら表示された統計（passages / sentences / entities / edges / 所要時間）を報告する

## ワークフロー2: 質問応答

```bash
uv run .claude/skills/linearrag/scripts/query_graph.py \
  --index linearrag_index/<名前> --query "<質問>" [--top-k 5]
```

出力JSONの読み方:
- `activated_entities`: 意味伝播が辿ったエンティティ連鎖。`level` は活性化した反復
  （1=クエリ直結、2以降=橋渡しエンティティ）。multi-hop の推論チェーンとして回答の根拠説明に使える
- `passages`: スコア順のtop-kパッセージ（rank / id / score / text）

**回答の生成手順:** passages の本文だけを根拠に回答する。根拠が不足していればその旨を述べる。
activated_entities の連鎖を「どう辿って見つけたか」の説明に使う。

## ワークフロー3: 検索品質の評価（オプション）

`{question, answer, evidence}` 形式の questions.json があれば:

```bash
uv run .claude/skills/linearrag/scripts/evaluate.py \
  --index linearrag_index/<名前> --questions <questions.json> [--limit N]
```

Retrieval-Contain（正解文字列がtop-k内に出現する率）と Evidence-Entity Recall を表示する。

## 調整パラメータ

| パラメータ | 既定値 | 効果 |
|---|---|---|
| --delta | 0.5 | 活性化の枝刈り閾値。上げると精度↑recall↓（test10では0.8でContain 8/10） |
| --sigma-top-n | 200 | 意味伝播に使うクエリ関連文の上限。伝播の暴走防止（0で無効） |
| --top-k | 5 | 返すパッセージ数 |
| --damping | 0.5 | PPRの減衰係数 |
| --lam / --w-p | 1.5 / 0.05 | 式7のDPR項とエンティティ項のバランス |
| --max-iterations | 4 | 意味伝播の最大hop数 |

参考実測（dataset/test10、658チャンク・10問、en_core_web_trf索引）:
既定値で Retrieval-Contain 7/10、--delta 0.8 で 8/10。

## トラブルシューティング

- **検索精度が低い**: まずエンティティ語彙を疑う。`entities.json` にクエリの固有名詞が
  入っているか確認し、無ければ `--spacy-model en_core_web_trf` で再構築
- **無関係なパッセージばかり返る**: 活性化エンティティのスコアが桁違いに大きい場合、
  "first" のような数量系ハブエンティティが暴れている可能性。クエリ側は
  CARDINAL/ORDINAL/DATE等を自動除外するが、必要なら --sigma-top-n を下げる
- **索引とクエリのモデル不整合**: クエリ側は索引の `meta.json` に記録されたモデルを
  自動使用するので、通常は起こらない

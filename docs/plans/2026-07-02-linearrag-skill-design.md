# LinearRAG スキル設計書

日付: 2026-07-02
対象論文: LinearRAG: Linear Graph Retrieval-Augmented Generation on Large-Scale Corpora (arXiv:2510.10114v4)
公式実装: https://github.com/DEEP-PolyU/LinearRAG

## 目的

論文 LinearRAG の「グラフ構築（Tri-Graph）」と「グラフRAG（2段階検索）」を、
Claude Code スキルとして Python で実装する。LLM トークン消費ゼロの索引構築と、
疎行列演算による高速な multi-hop 検索を再現する。

## 合意済みの設計判断

| 項目 | 決定 |
|---|---|
| スキル形態 | `.claude/skills/linearrag/` の **1スキル**に build / query 両スクリプトを同梱（自己完結、共通コード重複なし） |
| 対象言語 | 日英両対応（spaCy en_core_web_sm / ja_core_news_sm、多言語埋め込み） |
| 回答生成 | Python は top-k パッセージ返却まで。回答生成はスキルを呼んだ Claude が担当 |
| 入力形式 | ①ディレクトリ（.txt/.md 自動チャンク） ②JSONL（{id, title, text}） ③dataset 形式 chunks.json（"id:本文" 文字列リスト） |
| 依存管理 | PEP 723 インラインメタデータ + `uv run`（リポジトリの pyproject.toml は変更しない） |
| テストデータ | `dataset/test10/`（658チャンク・質問10件）を主に使用 |

## 構成

```
.claude/skills/linearrag/
├── SKILL.md              # 構築・検索両ワークフローのトリガー条件と使用手順
└── scripts/
    ├── build_graph.py    # グラフ構築 CLI（論文 §3.1）
    ├── query_graph.py    # 2段階検索 CLI（論文 §3.2）
    └── common.py         # 索引 I/O・埋め込みロード・言語判定
```

### 索引フォーマット（build の出力 = query の入力）

デフォルト出力先: `linearrag_index/<名前>/`

- `passages.jsonl` / `sentences.jsonl` / `entities.json` — 原文とノード定義
- `C.npz` — passage×entity contain 行列（scipy CSR、**出現回数**で保持。式7の N_ei に必要。二値化は使用時）
- `M.npz` — sentence×entity mention 行列（scipy CSR、二値）
- `emb_passages.npy` / `emb_sentences.npy` / `emb_entities.npy` — 埋め込み
- `meta.json` — 埋め込みモデル名・言語・パラメータ（query 側が整合性検証）

## build_graph.py（論文 §3.1: Token-Free Graph Construction）

CLI: `uv run build_graph.py --input <パス> --output linearrag_index/<名前> [--lang auto|en|ja] [--model <埋め込みモデル>]`

パイプライン（LLM 呼び出しゼロ）:
1. 入力形式を自動判別してパッセージ集合 P を得る
2. 文分割: 英語は句読点ベース、日本語は「。！？」ベース（spaCy sentencizer）
3. NER: spaCy でエンティティ抽出。小文字化・正規化で名寄せ（entity alignment 相当）
4. 疎行列 C（回数）と M（二値）を構築
5. passage / sentence / entity を sentence-transformers で一括エンコード
   - デフォルト: `paraphrase-multilingual-mpnet-base-v2`
   - `--model all-mpnet-base-v2` で論文設定を再現可能
6. 索引を保存し、統計（ノード数・エッジ数・所要時間）を表示

## query_graph.py（論文 §3.2: 2段階検索）

CLI: `uv run query_graph.py --index linearrag_index/<名前> --query "質問文" [--top-k 5] [--delta 0.4] [--json]`

**Stage 1 — エンティティ活性化**（式3〜5）:
1. クエリを spaCy で NER → 各クエリエンティティを索引内の最類似エンティティ
   （埋め込み cos 類似）にマッチし初期活性値を設定（式3）
2. クエリ×全文の類似度ベクトル σq を計算（式4）
3. `a_t = MAX(Mᵀ(σq ⊙ (M·a_{t-1})), a_{t-1})` を疎行列演算で反復（式5、最大4回）
4. 動的枝刈り: 新規活性エンティティのスコアが δ 未満なら棄却、新規ゼロで自動停止

**Stage 2 — パッセージ検索**（式6〜7）:
1. passage 初期スコア = `λ·sim(q,p) + ln(1 + Σ a_i·ln(1+N_ei)/L_ei)·W_p`（λ=0.05）
2. entity 初期スコア = Stage 1 の活性値。これらをシードに passage-entity 二部グラフで
   Personalized PageRank（damping 0.85、べき乗法）
3. 上位 k 件のパッセージを、活性化エンティティの反復履歴（推論チェーンの可視化、
   論文 Table 7 相当）と併せて JSON 出力 → Claude が読んで回答を生成

**ハイパーパラメータ既定値**: k=5、λ=0.05、damping=0.85、最大反復4

### 論文の曖昧点と対処

- **δ の値**: 本文は「δ=4」だが図5a の横軸は 0.2〜0.8 で矛盾 → デフォルト δ=0.4、`--delta` で調整可能
- **式7の L_ei（エンティティ階層レベル）と W_p（パッセージ重み係数）**: 本文に定義なし →
  実装時に公式リポジトリを参照して確定。参照不能なら中立値 L_ei=1、W_p=1 でスタート

## テスト・検証計画

**レベル1 — ユニットテスト**（pytest、極小合成コーパス）:
- C / M 行列構築の期待エッジ検証
- 式5の伝播で 2-hop 橋渡しエンティティが活性化されること
- 動的枝刈りの停止条件、PPR の収束

**レベル2 — test10 結合テスト**:
1. 索引構築（CPU で数分想定）→ ノード数・エッジ数・時間を確認
2. 10問で測定: Retrieval Contain（正解文字列が top-5 内に出現）、
   Evidence Recall（根拠エンティティが活性化リストに含まれる）
3. オプションで hotpotqa 等 1000 問セットでも計測

**レベル3 — E2E**: スキル経由で Claude が構築→検索→回答生成を通しで実行。

**成功基準**: test10 で Retrieval Contain 7/10 以上。下回る場合は δ や式7の
曖昧パラメータを公式リポジトリと突き合わせて調整。

**注意**: 初回実行時に埋め込みモデル（約1GB）と spaCy モデルのダウンロードが発生。

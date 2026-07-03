# LinearRAG 改良研究 — マスターノート（生きた文書）

> **これが唯一の正典。以後はこのファイルを更新していく。**
> 個別ノート（下記）は詳細の付録。数字・結論はここに集約する。
> 最終更新: 2026-07-03

LinearRAG（arXiv:2510.10114、関係フリー Tri-Graph 検索）を、**トークンフリー・訓練フリーのまま**
改良して論文化する共同研究。skill 化はしない（研究モード）。

### 付録（詳細ノート）
- 失敗分析: `2026-07-03-failure-analysis.md`
- Stage 1 方向・結果: `2026-07-03-research-direction.md` / `2026-07-03-resirag-results.md`
- 評価基盤: `2026-07-03-eval-basis.md`
- Stage 2 方向・全結果: `2026-07-03-stage2-coverage-direction.md`
- 旧サマリー: `2026-07-03-research-summary.md`（本ファイルに統合済み）

### コード（すべて元の skill を非破壊で拡張）
- `.claude/skills/linearrag/scripts/resirag.py` — Stage 1 改良（ResiRAG）＋ `stage2_candidates()`
  （Stage 2 実験用に内部量を返すフック）。`residual_strength=0` で元実装とビット一致。
- `research/build_gold_datasets.py` — 正典コーパス変換（gold_ids 付き）
- `research/eval_gold_recall.py` — gold-passage 評価器
- `research/rank_headroom.py` — Stage 2 伸び代探針
- `research/stage2_experiment.py` — Stage 2 選択戦略の比較（score/MMR/dense/cover/soft/residual + 対照 + `--s1`）
- `research/recall_curve.py` — LLM フリーの文脈効率（Recall@k・Precision@k・min-k）

---

## 0. 一言サマリー

LinearRAG の検索を 2 段階に分解して診断し、**ボトルネックは Stage 2（最終選択）にある**と gold 指標で
特定。**活性化被覆に基づくトークンフリーな選択（coverSoft）**で GoldRecall@5 を全データセットで改善
（+2.9〜+15.2pt）。対照実験で「効いているのは汎用の多様化でなくグラフ活性化信号」を実証。
Stage 1 改良（ResiRAG）は単独では小効果。現在 Stage 1×Stage 2 の合成を検証中。

---

## 1. LinearRAG の 2 段階（前提）

- **Stage 1 = エンティティ活性化**: 質問エンティティから、意味の近い文を伝って関連エンティティへ
  活性を広げる（式5、閾値 δ）。マルチホップの「橋渡し」。出力＝各エンティティの活性値・レベル。
- **Stage 2 = パッセージ検索**: 活性化を種に passage×entity グラフで PPR（式6-7）、上位 k を返す。
  **元の実装は「PPR スコア上位 5 件を 1 本ずつ点数順に切る」だけ**。

---

## 2. 全体の物語（4 幕）

**幕1 診断**: 失敗は二峰性 — under-propagation（活性化がシードで死ぬ 60%）と over-propagation
（活性化爆発 33%）。単一 δ では両立不能（δ を動かすと復活と喪失が相殺）。

**幕2 Stage 1 改良（ResiRAG）**: ホップごとに「カバー済みクエリ成分」を直交射影で引いた**残差クエリ**
で σ を測り直す＋相対閾値＋適応ゲート。→ **効果小**（AllGoldHit +0.3〜+3pt）。

**幕3 評価基盤整備＋ボトルネック特定**: gold-passage 指標を導入したら、Stage 1 を直しても伸びない理由が
判明 = **2 つ目の根拠が Stage 2 の点数順カットで落ちている**。ボトルネックは Stage 2 に移っていた。

**幕4 Stage 2 改良（本命）**: top-5 の 71〜87% が上位とエンティティ重複（hop1 の言い換えで埋まる）。
集合を意識した選択で 2 つ目の根拠を浮上。→ **coverSoft が GoldRecall を大きく改善**。

---

## 3. 評価基盤（信頼できる・検証済み）

- **正典コーパス**（HippoRAG reproduce と同じ検索プール）を `dataset/{2wiki,hotpot,musique}_hpr/` に。
  段落＝パッセージ、`gold_ids` をパッセージ単位で保持（musique の同一タイトル別段落も厳密同定）。
- **NER は trf に統一**（`linearrag_index/*_hpr_trf`）。sm→trf で GoldRecall 2wiki +11.8pt。
  エンティティ数はほぼ同じでも質（境界・リンク）が支配的。
- **指標**: Contain@5（表層一致・参考）/ **GoldRecall@5** / **AllGoldHit@5**（全根拠 top-5、主指標）。
- **検証済み**: 索引 id == corpus 行、musique gold 100% 本文一致、end-to-end 経路（recall@50≠0）。
- **決定版 baseline（trf, full 1000, = 元 LinearRAG）**:

  | データセット | Contain@5 | GoldRecall@5 | AllGoldHit@5 |
  |---|---|---|---|
  | 2wiki | 53.5% | 63.1% | 35.2% |
  | hotpot | 54.7% | 51.8% | 29.5% |
  | musique | 39.6% | 37.4% | 11.0% |

---

## 4. Stage 2 選択戦略（本丸）— 結果まとめ

### 4.1 選択戦略の比較（trf, 300q, top_n=50, GoldRecall / AllGoldHit）

| 選択戦略 | 2wiki | musique | hotpot |
|---|---|---|---|
| score（元 LinearRAG＝点数順） | 63.5 / 36.3 | 37.8 / 9.7 | 53.5 / 31.3 |
| **coverSoft（最良, γ=0.5）** | **67.0 / 36.3** | **44.8 / 10.3** | **68.8 / 43.0** |
| coverDense（飽和なし） | 66.4 / 35.3 | 43.9 / 10.3 | 67.5 / 40.3 |
| dense（密のみ） | 54.0 / 22.7 | 37.1 / 7.7 | 65.3 / 38.3 |
| MMR（元 PPR + 埋め込み多様性） | 56.0 / 24.7 | 31.9 / 2.3 | 46.2 / 21.7 |
| oracle（上限） | 80.8 / 59.0 | 69.7 / 40.7 | 88.0 / 76.3 |

### 4.2 coverSoft@0.5 の baseline 比（full 1000q, 確定値）

| データセット | GoldRecall@5 | AllGoldHit@5 |
|---|---|---|
| 2wiki | 63.1 → **66.0**（+2.9） | 35.2 → 34.8（−0.4） |
| musique | 37.4 → **44.5**（+7.1） | 11.0 → **12.0**（+1.0） |
| hotpot | 51.8 → **67.0**（+15.2） | 29.5 → **42.3**（+12.8） |

### 4.3 文脈効率（LLM フリー・「ゴミの少なさ」, 400q）

| 指標 | 2wiki | musique | hotpot |
|---|---|---|---|
| Precision@1 score→coverSoft | 74.5→**86.5** | 47.8→**57.2** | 40.8→**60.8** |
| Recall@3 score→coverSoft | 57.2→**62.9** | 30.2→**38.9** | 42.2→**58.2** |
| min-k(all gold) 中央値 | 4→**3** | 17→**12** | 7→**5** |

先頭チャンクが gold の率が上がり（トップのゴミ減）、両根拠を揃えるのに読む量が減る。

---

## 5. 手法 coverSoft の中身

元の Stage 2「点数順 top-5」を、後段の選択レイヤーに差し替え（構築・活性化・PPR は不変、
新モデル・LLM・訓練なし）:
1. **候補バッファ**: PPR 上位 N=50 を残す。
2. **密アンカー**: 並べ替え基準を PPR スコアから密類似度 `sim_qp`（既計算）へ。
   （見逃し gold は密では上位＝PPR が沈めている、という診断に基づく）
3. **活性化 IDF 飽和被覆の貪欲選択**:
   `gain(p|S) = 密類似度(p) + Σ_{e∈p} activation[e]·IDF[e]/level[e] · γ^(eの被覆回数)`
   希少エンティティ（橋・答え側）を重視、遍在シードを軽視。γ≈0.5 の飽和で**完成ペアを壊さない**。

すべて LinearRAG が既に持つ量（activation, level, sim_qp, C）の再利用＝**増分索引の軽さを保つ**。

---

## 6. 確定した知見（論文の主張候補）

1. **失敗は二峰性、δ では両立不能**（介入実験で裏付け）。
2. **ボトルネックは Stage 1 でなく Stage 2**（gold 指標で発見）。AllGoldHit が律速。
3. **MMR（汎用多様化）はマルチホップに逆効果**（gold 同士が意味的に近く、多様性が 2 つ目を追い出す）。
4. **効くのは汎用の重複除去でなくグラフ活性化被覆**: 対照実験で
   **coverDense > coverUniform > coverAll**（汎用エンティティ除去 coverAll は最悪）。
5. **coverSoft（飽和被覆）が完成ペアの引き剥がしを解消**（恣意的ランク固定なし、γ 1 個）。
6. **統一原理（残差）は entity 粒度でのみ有効、passage 粒度では不成立**（Stage 2 残差 C は失敗）。
7. **文脈効率も改善**（Precision@1・min-k）＝ retrieval の主張は e2e 不要、LLM フリーで閉じる。
8. **増分索引が軽い**: 関係なし・LLM なし・大域要約なしのため、新規データ追加が append 主体。
   coverSoft は索引に何も足さない（クエリ時のみ）＝この利点を保存。

---

## 7. 論文の枠組み

- **主語**: 「新しい選択アルゴリズムの発明」ではなく、
  「**トークンフリー関係フリー GraphRAG には診断可能な Stage 2 ボトルネックがあり、グラフ自身の
  クエリ活性化を飽和被覆信号として使うと、汎用の多様化が失敗・逆効果になる場面でそれが解ける**」。
- **非自明さ = トークンフリー制約**: LLM-RAG なら LLM リランカーで済むため誰も探さない問題。
  制約が移植を非自明にし、移植が新発見（MMR 逆効果・活性化被覆・残差の粒度依存）を生んだ。
- **貢献**: (a) 二峰性診断と δ 限界、(b) gold-passage 評価基盤＋日本語マルチホップ、
  (c) 活性化被覆 Stage 2（対照でグラフ信号の寄与を実証）、(d) 否定的知見（MMR/残差）、
  (e) 増分索引の運用優位。
- **差別化**: PropRAG(LLM 命題)/BridgeRAG(LLM judge)/BDTR(LLM 思考) は全て LLM 依存。
  本手法はトークンフリー・訓練フリー・レイテンシ増ほぼゼロ・増分追加が軽い。
- **指標哲学**: e2e は reader LLM を交絡させるので主指標にしない。gold Recall/AllGoldHit ＋
  文脈効率（Precision@1・min-k）で retrieval を閉じて評価。e2e はやるとしても安全弁として最小限。
- **投稿前に潰す**: (a) 要約の劣モジュラ被覆（Lin-Bilmes 系）、(b) 多様化 PPR / coverage-based
  selection、(c) SPRIG（arXiv:2602.23372, 同路線）との重複。

---

## 8. 現在地と次の手

### 完了
- 失敗診断、評価基盤（正典・trf・gold_ids・検証）、baseline 確定。
- Stage 1（ResiRAG）実装・評価（効果小）。
- Stage 2: coverDense→coverSoft、対照実験（新規性証明）、full 1000 確定、文脈効率。

### Stage 1 × Stage 2 合成の結論（300q, GoldRecall / AllGoldHit）

| データセット | baseline-S1 + coverSoft | ResiRAG-S1 + coverSoft | ResiRAG-S1 + score |
|---|---|---|---|
| 2wiki | 67.0 / 36.3 | 63.1 / 33.0 | 56.6 / 34.0 |
| musique | 44.4 / 10.3 | 44.4 / **13.3** | 37.0 / **14.7** |
| hotpot | 68.5 / 42.7 | 67.2 / 40.3 | 52.7 / 29.0 |

- **普遍的な相乗はない**。2wiki・hotpot は ResiRAG Stage 1 が悪化（baseline が強く、活性化変更が候補を濁す）。
- **musique でのみ ResiRAG Stage 1 が AllGoldHit を改善**（score 9.7→14.7, coverSoft 10.3→13.3）＝
  under-prop が支配的な難データ＝**ResiRAG の本来の主戦場でだけ効く**。失敗診断と完全整合。
- **含意**: 「積み上がる単一システム」ではなく「**二つの改良が別の失敗モードに対応**」という物語。
  既定は baseline-S1 + coverSoft、under-prop 支配データでは ResiRAG-S1 を選ぶ（データ依存）。

### 論文としての現実的評価（2026-07-03 時点の自己評価）

- **トップ会議のメソッド論文としては弱い**: 手法新規性が薄い（被覆リランキングは既知）、主指標
  AllGoldHit が動きにくい（hotpot 以外）、効果の一部が dense 由来、SoTA に未達、合成が拮抗。
- **強み**: 診断の質（二峰性・δ限界・Stage2 特定・対照実験・否定的知見）、トークンフリー/増分索引の
  実用ニッチ、日本語マルチホップ（未着手）。
- **着地**: Findings / ワークショップ / ショートは射程。本会議に上げるレバー →
  (1) 日本語マルチホップを当てる、(2) **coverSoft を LinearRAG 以外にも載せて「汎用トークンフリー
  選択」に昇格**（incremental 批判を一般性で封じる、最有効）、(3) 主語を GoldRecall＋効率に寄せる、
  (4) e2e を 1 点だけ安全弁に、(5) 増分索引のマイクロベンチ。

### 次
1. **JEMHopQA（日本語マルチホップ）を gold_ids 方式で評価**（差別化の一枚看板・レバー1）。
2. **JEMHopQA（日本語マルチホップ）を gold_ids 方式で評価**（英語圏の先行がやっていない差別化）。
3. γ の最終決定、α・top_n の感度。
4. （欲を言えば）増分索引のマイクロベンチ（追加コスト vs LLM 系）を 1 つ。
5. 関連研究の最終サーベイ（§7 の潰しリスト）。

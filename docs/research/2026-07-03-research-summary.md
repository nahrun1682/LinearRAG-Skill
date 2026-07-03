# LinearRAG 改良研究 — ここまでのまとめ（2026-07-03）

LinearRAG（arXiv:2510.10114、関係フリー Tri-Graph 検索）を改良して論文化を目指す共同研究の
進捗サマリー。個別ノートへのハブとして使う。skill 化はしない（研究モード）。

- 失敗分析 → [2026-07-03-failure-analysis.md](2026-07-03-failure-analysis.md)
- 研究方向（Stage 1 残差伝播）→ [2026-07-03-research-direction.md](2026-07-03-research-direction.md)
- ResiRAG(Stage 1) 結果 → [2026-07-03-resirag-results.md](2026-07-03-resirag-results.md)
- 評価基盤 → [2026-07-03-eval-basis.md](2026-07-03-eval-basis.md)
- Stage 2 方向＋結果 → [2026-07-03-stage2-coverage-direction.md](2026-07-03-stage2-coverage-direction.md)

コード（すべて元の skill を変更せず別モジュール）:
- `.claude/skills/linearrag/scripts/resirag.py` — Stage 1 改良＋Stage 2 実験用フック
- `research/build_gold_datasets.py` — 正典コーパス変換（gold_ids 付き）
- `research/eval_gold_recall.py` — gold-passage 評価器
- `research/rank_headroom.py` — Stage 2 伸び代探針
- `research/stage2_experiment.py` — Stage 2 選択戦略の比較

---

## 1. 全体の物語（3 幕）

**幕1: 診断**。LinearRAG の検索失敗を分類したら二峰性だった —
under-propagation（活性化がシードで死ぬ、60%）と over-propagation（活性化が爆発、33%）。
単一の閾値 δ ではどちらも直せない（δ を変えると復活と喪失が相殺）。

**幕2: Stage 1 改良（ResiRAG）**。各ホップでカバー済みクエリ成分を引いた「残差クエリ」で
σ を測り直す＋相対閾値＋適応ゲート。**元実装とビット一致（residual_strength=0）** を保証。
→ 効果は小さかった（AllGoldHit +0.3〜+3pt）。gold-passage 指標を導入して理由が判明:
**ボトルネックは Stage 1 でなく Stage 2（最終ランキング）に移っていた**。

**幕3: Stage 2 改良（本命・進行中）**。top-5 の 71〜87% が上位とエンティティ重複（hop 1 の
言い換えで埋まる）。集合を意識した選択で 2 つ目の根拠を浮上させる。
→ **coverDense**（密類似度アンカー＋活性化 IDF 被覆）が GoldRecall@5 で baseline を
一貫して上回り（+2.9〜+14pt）、対照の埋め込み MMR を大きく超えた（＝グラフ信号が効く証明）。
現在 full 1000 で確定中。

## 2. 確定した知見

1. **失敗は二峰性、δ では両立不能**（介入実験で裏付け済み）。
2. **評価基盤を trf 統一・正典コーパス・gold_ids に整備**（下記 §3）。ここが研究の土台。
3. **NER 品質が支配的**: sm→trf で GoldRecall 2wiki +11.8pt。エンティティ数はほぼ同じでも
   質（境界・リンク）が効く。
4. **マルチホップの律速は AllGoldHit**（全根拠を揃える）。GoldRecall(片方は取れる)との差が
   「マルチホップの壁」。Stage 2 の点数順カットが集合の相補性を見ていないのが構造的原因。
5. **MMR（汎用多様化）はマルチホップに逆効果**（gold 同士が意味的に近く、多様性が 2 つ目を追い出す）。
6. **当初案（PPR 上の活性化被覆）は不発**。見逃し gold の 47% は識別特徴が「答え」で未活性化。
7. **効いた形＝ coverDense**。見逃し gold は密類似度では上位（PPR が沈めている）という診断から、
   密アンカー＋活性化被覆に修正して当てた。**対照実験（vs denseMMR）でグラフ信号の寄与を実証**。

## 3. 評価基盤（信頼できる、検証済み）

- **正典コーパス**（HippoRAG reproduce と同じ検索プール）を `dataset/{2wiki,hotpot,musique}_hpr/` に。
  段落＝パッセージ、`gold_ids` をパッセージ単位で保持（musique の同一タイトル別段落も厳密同定）。
- **trf 索引** `linearrag_index/*_hpr_trf`（en_core_web_trf）。
- **指標**: Contain@5 / GoldRecall@5 / **AllGoldHit@5**（全根拠 top-5、主指標）。
- **検証済み**: id 対応（索引 id == corpus 行）、musique gold 100% 本文一致、end-to-end 経路。
- **決定版 baseline（trf, full 1000, = 元 LinearRAG）**:

  | データセット | Contain@5 | GoldRecall@5 | AllGoldHit@5 |
  |---|---|---|---|
  | 2wiki | 53.5% | 63.1% | 35.2% |
  | hotpot | 54.7% | 51.8% | 29.5% |
  | musique | 39.6% | 37.4% | 11.0% |

## 4. 現在の最良結果（Stage 2, trf, 300q, top_n=50）

| 選択戦略 | 2wiki R/AGH | musique R/AGH | hotpot R/AGH |
|---|---|---|---|
| score（元 LinearRAG） | 63.5 / 36.3 | 37.8 / 9.7 | 53.5 / 31.3 |
| **coverDense（提案）** | **66.4 / 35.3** | **43.9 / 10.3** | **67.5 / 40.3** |
| denseMMR（対照） | 53.8 / 20.0 | 36.0 / 5.7 | 64.7 / 37.7 |
| oracle（上限） | 80.8 / 59.0 | 69.7 / 40.7 | 88.0 / 76.3 |

（R=GoldRecall@5, AGH=AllGoldHit@5。full 1000 で確定作業中）

## 5. coverDense が LinearRAG に足しているもの

最終選択ステップだけ差し替え（構築・活性化・PPR は不変）。新モデル・LLM・訓練なし:
1. **候補バッファ**: top-5 即切りをやめ PPR 上位 N=50 を残す。
2. **密アンカー**: 並べ替え基準を PPR スコアから密類似度 `sim_qp`（既計算）へ。
3. **活性化 IDF 被覆の貪欲選択**: `gain = 密類似度 + Σ_{未被覆な活性化エンティティ} activation·IDF/level`。
   希少エンティティ（橋・答え側）を重視、遍在シードを軽視 → 冗長を排して 2 つ目の根拠を浮上。

すべて LinearRAG が既に持つ量（activation, level, sim_qp, C 行列）の再利用。

## 6. 論文の骨子（想定）

- **タイトル案**: トークンフリーな関係フリー GraphRAG の多ホップ検索を、状態条件付き選択で強化。
- **貢献**: (a) 二峰性の失敗診断と δ の限界、(b) gold-passage 評価基盤 ＋ 日本語マルチホップ、
  (c) 活性化被覆による Stage 2 選択（対照実験でグラフ信号の寄与を実証）、(d) MMR がマルチホップに
  逆効果という否定的知見。
- **一貫原理**: 「検索の状態で次の判断を条件付ける」（Stage 1 = 残差クエリ、Stage 2 = 被覆集合）。
- **差別化**: PropRAG(LLM 命題)/BridgeRAG(LLM judge)/BDTR(LLM 思考) は全て LLM 依存。
  本手法はトークンフリー・訓練フリー・レイテンシ増ほぼゼロ。

## 7. 残課題・次の手

1. **full 1000 で coverDense の勝ちを確定**（進行中）。
2. **AllGoldHit のまだら**を改善（2wiki −1）。α チューニング、候補生成を密にする ablation。
3. **Stage 1(ResiRAG) × Stage 2(coverDense) の合成**で相乗効果を見る。
4. **JEMHopQA（日本語）** を ids 方式に寄せて同じ土俵で評価 ＝ 英語圏の先行がやっていない差別化。
5. **効果の分解**（密アンカー ② vs 活性化被覆 ③ の寄与切り分け。現状 coverDense >> dense なので ③ が主役）。

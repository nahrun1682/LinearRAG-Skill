# LinearRAG 改良研究 — マスターノート（生きた文書）

> **これが唯一の正典。以後はこのファイルを更新していく。**
> 冷えた状態で読んでも、やってきたこと・仕組み・数字・各判断の理由が再構成できることを目標に書く。
> 個別ノートは詳細の付録。数字と結論はここに集約する。
> 最終更新: 2026-07-03

---

## 目次
0. プロジェクトの目的と制約
1. 用語・指標の定義（先に読む）
2. LinearRAG の仕組み（改良対象の理解）
3. 評価基盤（土台。ここが信頼できないと全部無意味）
4. 研究の物語（時系列＋各分岐の理由）
5. 我々が作った手法（ResiRAG / coverDense / coverSoft）の詳細
6. 全実験と数字
7. 確定した知見（肯定的・否定的）
8. 論文としての枠組みと自己評価（正直版）
9. コード資産の目録
10. 現在地と次の手
11. 再現コマンド集

---

## 0. プロジェクトの目的と制約

- **目的**: LinearRAG（arXiv:2510.10114、関係フリー Tri-Graph 検索）を改良し、論文化する。
- **絶対制約**: **トークンフリー・訓練フリーを保つ**（LLM 呼び出しゼロ、モデル再学習なし）。
  これが本研究の存在理由。LLM を使ってよいなら LLM リランカーを 1 回呼べば済む問題が多い。
- **モード**: 研究。skill 化はしない。元の skill コード（`.claude/skills/linearrag/scripts/`）は
  **非破壊**で、改良は別モジュール（`resirag.py`）や `research/` に置く。
- **検証環境**: WSL、`uv run`（PEP 723 インライン依存）。索引は `linearrag_index/`、データは `dataset/`。

---

## 1. 用語・指標の定義（先に読む）

**Tri-Graph**: LinearRAG の索引。3 種のノード（パッセージ / 文 / エンティティ）と、関係ラベルを
持たない帰属エッジだけからなる。行列で表現:
- **C 行列**（パッセージ × エンティティ、出現回数）: パッセージ p がエンティティ e を何回含むか。
- **M 行列**（文 × エンティティ、0/1）: 文 s がエンティティ e に言及するか。

**活性化 (activation)**: クエリのエンティティを起点に、グラフ上で各エンティティに割り当たる
スコア。Stage 1 の出力。`level`（何ホップ目で活性化したか、L_ei）も付く。

**PPR (Personalized PageRank)**: 活性化を種に、パッセージ×エンティティ二部グラフ上を回す
ランダムウォーク。パッセージの最終スコアを与える（Stage 2）。

**指標（すべて top-5、正解の"根拠パッセージ"に対して）**:
- **Contain@5**: top-5 の本文に答え文字列（＋別名）が出現するか。**表層一致**なので、言い換えで
  偽陰性、一般語（日付等）で偽陽性。参考指標。
- **GoldRecall@5**: 各質問の gold パッセージ集合のうち top-5 に入った割合の平均（部分点あり）。
- **AllGoldHit@5**: **全ての** gold が top-5 に揃った質問の割合（全か無か）。マルチホップの主指標。
  「両方の根拠を揃えて初めて答えられる」を測る。**GoldRecall > AllGoldHit** が常に成立。
- **Precision@1 / min-k**: 文脈効率（§6.5）。先頭が gold の率／全 gold 揃うまでに読む数。

**gold パッセージ**: データセットが「この質問の根拠」と注釈したパッセージ。パッセージ ID で保持
（`gold_ids`）。マルチホップなので 1 質問に 2〜4 個。

---

## 2. LinearRAG の仕組み（改良対象の理解）

検索は 2 段階。比喩: Stage 1 =「頭の中で関連キーワードを芋づるに広げる」、
Stage 2 =「そのキーワードを一番よく含む本を本棚から選ぶ」。

### Stage 1 — エンティティ活性化（式3-5）
1. クエリから NER でエンティティを抽出（例:「坊っちゃん」）。数値ラベル（日付等）は除く。
2. 各クエリエンティティを、埋め込みが最も近いグラフエンティティに初期活性化（式3）。
3. **伝播（式5）**: 活性化を、そのエンティティが登場する文へ流し（M）、クエリと文の類似度 σ で
   重み付けし、同じ文の他エンティティへ戻す（Mᵀ）。式:
   `a_t = max( Mᵀ(σ ⊙ (M a_{t-1})), a_{t-1} )`
   新たに活性化するのは値が閾値 **δ** を超えたものだけ。数ホップ繰り返す。
   - σ = クエリ埋め込みと各文埋め込みのコサイン類似度（＝「文の通りやすさ」）。
   - δ = 「活性化と認める最低ライン（堰の高さ）」。高いほど広がりにくい。
   - **重要な弱点**: σ はクエリ**全体**との類似度で**最初に 1 回だけ**決めて固定。
     「今どのホップにいて、何をカバー済みか」を持たない。

### Stage 2 — パッセージ検索（式6-7）
1. 活性化エンティティ＋密類似度項を種に、パッセージ×エンティティ二部グラフで PPR（式6）。
2. 式7 で各パッセージの初期スコア: `(λ·minmax(sim_qp) + ln(1+Σ a_i·ln(1+N_pi)/L_i))·W_p`
   （sim_qp = クエリ-パッセージ密類似度、N_pi = 出現回数、L_i = 活性化レベル）。
3. **PPR スコアの上位 5 件を、1 本ずつ点数順に切って返す**。
   - **重要な弱点**: 集合の相補性を見ない。top-5 が「同じ話の言い換え」で埋まりうる。

### パラメータ既定値
δ=0.8（2wiki でチューニング、参照実装は 0.5）、λ=1.5、W_p=0.05、damping=0.5、
max_iterations=4、sigma_top_n=200（σ を上位 200 文に疎化）、top_k=5。

---

## 3. 評価基盤（土台）

**なぜ作り直したか**: 当初の `dataset/*/chunks.json` は小さく・タイトル欠落・小文字化された
非正典コーパスで、gold 注釈も落ちていた。数値が literature と比較不能で、gold-passage 評価も不可能。

**何をしたか** (`research/build_gold_datasets.py`):
- **正典コーパス**を HippoRAG `reproduce/dataset/` から取得（HippoRAG2/PropRAG/BridgeRAG と
  **同じ検索プール**＝数値が直接比較できる）。`dataset/{2wiki,hotpot,musique}_hpr/` に変換。
- 段落＝パッセージ（チャンク分割なし＝ gold 対応が完全保存）。`gold_ids` を**パッセージ単位**で保持。
  - 2wiki/hotpot: gold = supporting_facts のタイトル → コーパス ID（1 タイトル 1 段落なので一意）。
  - musique: gold = is_supporting 段落。**同一タイトルに複数根拠段落**があるので (title,text) で厳密同定。
  - `answer_aliases` も保持（Contain の偽陰性低減）。
- コーパス規模: 2wiki 6,119 / hotpot 9,811 / musique 11,656 パッセージ。gold/質問 2.0〜2.65 個。

**NER は trf に統一**（決定的な較正）:
- `en_core_web_sm`（速い）vs `en_core_web_trf`（遅い、高品質）を 2wiki で比較 →
  **GoldRecall 51.2%(sm) → 63.5%(trf)、+12.3pt**。エンティティ数はほぼ同じ（39.8k vs 40.5k）
  なのに質（境界・リンクの正しさ）が支配的。→ **索引は `*_hpr_trf` に統一**。構築各 17〜33 分。

**評価器** (`research/eval_gold_recall.py`): Contain@5 / GoldRecall@5 / AllGoldHit@5 を出力。
strategy=ids（gold_ids 直読み）。

**検証済み（gold 評価が正しいことの確認）**:
- 索引の passage["id"] == corpus 行番号、本文も完全一致（全件）。
- musique の gold 同定は 2648 件が 100% 本文一致（タイトルフォールバック 0、取り違え 0）。
- gold_id のタイトルが元データの supporting タイトルと一致（同一タイトル 2 段落ケース含む）。
- headroom 探針で recall@50 が 0 でなく 67% ＝ 索引→検索→評価の id 経路が通っている傍証。

**決定版 baseline（trf, full 1000q, = 元 LinearRAG）**:

| データセット | Contain@5 | GoldRecall@5 | AllGoldHit@5 |
|---|---|---|---|
| 2wiki | 53.5% | 63.1% | 35.2% |
| hotpot | 54.7% | 51.8% | 29.5% |
| musique | 39.6% | 37.4% | 11.0% |

（参考: baseline LinearRAG は PropRAG 等の重量級 SoTA（R@5 ~0.95）より大きく低い。
軽量・トークンフリーの代償。だから伸び代 = headroom が大きい。）

---

## 4. 研究の物語（時系列＋各分岐の理由）

### 幕1: 失敗診断 — 二峰性
MuSiQue の検索失敗を、Retriever が返すトレース（query_entities / activated_entities / passages）と
コーパス全体の answer-string 含有を突き合わせて 4 分類（`research`→当時は `/tmp`）:
- **NER**（クエリからエンティティが取れない）7%
- **ACTIVATION / under-propagation**（活性化がシードで死ぬ、gold のエンティティが 1 つも活性化せず）**60%**
- **PRUNING / over-propagation**（活性化が数百に爆発、gold が薄まる）33%
- NOT_IN_CORPUS（答え文字列がコーパスに無い＝ Contain 指標のノイズ）は母数外

**δ では両立不能を実証**: under-propagation 失敗を δ を変えて再検索すると、復活率は
0%(δ0.8) → 36%(δ0.3) → 28%(δ0.1) と**非単調**（下げると渡れ始めるが下げすぎると爆発）。
全問での大域スイープでも「δ0.8→0.3 で +16 復活 / −12 新規喪失」＝約 14% が閾値で反転。
**単一の大域 δ では under と over を両立できない** → 適応が要る、という診断。

### 幕2: Stage 1 改良（ResiRAG）— 効果小
診断「σ が固定でホップ状態を持たない」に対し、**残差クエリ伝播**を実装（§5.1）。
→ gold 指標で測ると **AllGoldHit +0.3〜+3pt と小さい**。

### 幕3: 評価基盤整備 → ボトルネックが Stage 2 と判明
ResiRAG が伸びない理由を gold 指標で追ったら、**GoldRecall は高いのに AllGoldHit が低い**
（例: 2wiki GoldRecall 63% vs AllGoldHit 35%）。「片方の根拠は取れるが両方は揃わない」。
→ **headroom 探針**（`research/rank_headroom.py`, top-50 まで掘って gold の順位を見る）で:
- 見つかった gold の最悪順位の中央値 = 2wiki 4 / musique 16。**2 つ目の gold は rank 6〜20 に埋もれている**。
- top-5 の **71〜87% が上位とエンティティを共有**（hop1 の言い換えで埋まり、hop2 を押し出す）。
- 完璧な top-20→top-5 再ランカーの上限: AllGoldHit を 2wiki 36→49%、musique 10→23%、hotpot 31→55% に。
- **結論: ボトルネックは Stage 1 でなく Stage 2 の点数順カット**。ここに伸び代がある。

### 幕4: Stage 2 改良（本命）— coverDense → coverSoft
- 最初の案「PPR 上の活性化被覆」(cover) は **baseline と同点で不発**。
  診断で理由判明: 見逃し gold の 47% は「未被覆の活性化エンティティが 0」＝識別特徴が**答え**
  （クエリ時に未活性化）。活性化信号では距離が測れない。
- しかし見逃し gold は**密類似度では 77 パーセンタイル**（PPR が不当に沈めている）→ 設計変更:
  **密アンカー ＋ 活性化 IDF 被覆** = **coverDense**。GoldRecall が baseline を全データセットで上回る。
- **対照実験で新規性を証明**（幕4の核）: coverDense > coverUniform > coverAll（§6.3）。
  「効くのは汎用の重複除去でなくグラフ活性化被覆」を実証。MMR は逆効果。
- **完成ペア引き剥がし問題**（2wiki の AllGoldHit 微減）を診断し、**飽和被覆 = coverSoft** で修正（§5.3, §6.4）。
- full 1000 で確定、文脈効率（Precision@1・min-k）も改善（§6.5）。

### 幕5: Stage 1 × Stage 2 合成 — データ依存、普遍的相乗なし
ResiRAG-S1 + coverSoft を測ると、**musique（under-prop 支配）でだけ ResiRAG が AllGoldHit を改善**
（score 9.7→14.7）、2wiki/hotpot は悪化。→「積み上がる単一システム」でなく
「**2 つの改良が別々の失敗モードに対応**」。診断（ResiRAG は under-prop を直す道具）と整合。

---

## 5. 我々が作った手法の詳細

### 5.1 ResiRAG（Stage 1、`resirag.py`）
各ホップ t で、これまで活性化した上位 k エンティティの張る部分空間を、クエリから**直交射影で除去**した
**残差クエリ**で σ を測り直す。
```
q_{t+1} = normalize( q_t − strength · proj_{span(covered entities)}(q_t) )
σ_t(s) = cos(emb_s, q_{t+1})   # ホップごとに σ を更新
```
加えて **相対閾値**（絶対 δ でなく「そのホップ最大活性の delta_rel 倍」）と、**適応ゲート**
（活性化の広がりに応じて残差強度を調整。枯れたホップは強度 0）。
- **設計契約**: `residual_strength=0 かつ threshold_mode=absolute` で**元実装とビット一致**（検証済み）。
  これにより A/B が公平（baseline 行が正しく「素の LinearRAG」）。
- 判明した性質: 残差は「渡りすぎ(over)」を焦点化し、相対閾値が「渡れない(under)」を緩和。
  両者は別モードを担当し、素朴に足すと打ち消す。適応ゲートで両立を図る。効果は限定的。

### 5.2 coverDense（Stage 2）
元の「点数順 top-5」を、後段の選択レイヤーに差し替え。3 要素:
1. **候補バッファ**: PPR 上位 N=50 を残す。
2. **密アンカー**: 並べ替え基準を PPR スコアから密類似度 sim_qp（既計算）へ。
3. **活性化 IDF 被覆の貪欲選択**:
   `gain(p|S) = 密類似度(p) + Σ_{e∈p, e が S に未被覆} activation[e]·IDF[e]/level[e]`
   希少エンティティ（橋・答え側）を重視、遍在シード（hop-0、全候補に出る）を IDF で軽視。
   単調劣モジュラなので貪欲法に (1−1/e) 近似保証。

### 5.3 coverSoft（Stage 2、最終採用）
coverDense の**ハード被覆を飽和（ソフト）被覆に**する。エンティティ e を m 回目に覆う利得を
`w_e · γ^m`（γ≈0.5）に。
- **狙い**: 橋エンティティを最初の gold で覆っても、2 つ目の gold で**部分点を残す**ので、
  「橋を共有する 2 つ目 gold」が「新エンティティを覆う distractor」に枠を奪われない
  ＝**完成ペアの引き剥がしを防ぐ**。
- γ=0 でハード被覆（= coverDense）、γ=1 で無被覆。中間が飽和。γ=0.3/0.5 で安定（脆くない）。
- **恣意的なランク固定（top-1,2 を守る等）を使わず**、γ 1 個の原理的変更で機序を直撃。

**すべて LinearRAG が既に持つ量の再利用**（activation, level, sim_qp, C）。索引に何も足さない
（クエリ時のみ）＝ 増分索引の軽さを保存（§7-8）。

### 5.4 試して捨てた手法（否定的知見）
- **MMR**（埋め込みコサインで多様化）: マルチホップに**逆効果**。gold 同士が意味的に近く、
  多様性が 2 つ目を追い出す。
- **coverDual**（PPR+密の二重アンカー）: PPR の弱さを持ち込んで微減。不採用。
- **residual（Stage 2 残差スロット選択）**: 「Stage 1 と同じ残差演算を passage 粒度で」という
  美しい統一原理案。**passage 丸ごとの直交射影は粗すぎて関連まで消し、全滅**。強度を緩めても不成立。
  → **残差は entity 粒度でのみ効き、passage 粒度では成立しない**という知見。

---

## 6. 全実験と数字

### 6.1 決定版 baseline（trf, full 1000）→ §3 の表

### 6.2 選択戦略の比較（trf, 300q, top_n=50, GoldRecall / AllGoldHit）
| 選択戦略 | 2wiki | musique | hotpot |
|---|---|---|---|
| score（元 LinearRAG） | 63.5 / 36.3 | 37.8 / 9.7 | 53.5 / 31.3 |
| **coverSoft γ=0.5（採用）** | **67.0 / 36.3** | **44.8 / 10.3** | **68.8 / 43.0** |
| coverDense（飽和なし） | 66.4 / 35.3 | 43.9 / 10.3 | 67.5 / 40.3 |
| dense（密のみ） | 54.0 / 22.7 | 37.1 / 7.7 | 65.3 / 38.3 |
| MMR | 56.0 / 24.7 | 31.9 / 2.3 | 46.2 / 21.7 |
| oracle（上限） | 80.8 / 59.0 | 69.7 / 40.7 | 88.0 / 76.3 |

### 6.3 新規性の対照実験（trf, 300q, GoldRecall、同じ密アンカー、被覆信号だけ変える）
| 被覆信号 | 2wiki | musique | hotpot |
|---|---|---|---|
| **coverDense（活性化×IDF）** | **66.4** | **43.9** | **67.5** |
| coverUniform（活性化・均等重み） | 63.0 | 40.0 | 53.8 |
| coverAll（全エンティティ・均等＝汎用除去） | 35.3 | 27.1 | 42.8 |
- 単調に coverDense > coverUniform > coverAll。汎用除去(coverAll)が最悪＝「any dedup works」を否定。
- 活性化への限定で激変（2wiki +27.7pt）、活性化重みでさらに上乗せ（hotpot +13.7pt）。
- 「ただ dense が強い」も否定: dense 単体は 2wiki/musique で baseline に負けるが coverDense は勝つ。

### 6.4 完成ペア修正 A/B/C（trf, 300q）
| 手法 | 2wiki R/AGH | musique R/AGH | hotpot R/AGH |
|---|---|---|---|
| **A: coverSoft γ=0.5** | **67.0 / 36.3** | **44.8 / 10.3** | **68.8 / 43.0** |
| B: coverDual | 65.7 / 34.7 | 41.9 / 9.3 | 62.8 / 38.3 |
| C: residual@0.3 | 56.3 / 23.7 | 37.0 / 6.0 | 66.8 / 39.3 |
- A 採用（2wiki AllGoldHit 退行が 300q で解消、hotpot AGH +2.7）。B 不採用。C 棄却（§5.4）。

### 6.5 coverSoft@0.5 の baseline 比（full 1000q, 確定）＋ 文脈効率（400q）
| データセット | GoldRecall@5 | AllGoldHit@5 | Precision@1 | min-k中央値 |
|---|---|---|---|---|
| 2wiki | 63.1 → **66.0**（+2.9） | 35.2 → 34.8（−0.4） | 74.5→**86.5** | 4→**3** |
| musique | 37.4 → **44.5**（+7.1） | 11.0 → **12.0**（+1.0） | 47.8→**57.2** | 17→**12** |
| hotpot | 51.8 → **67.0**（+15.2） | 29.5 → **42.3**（+12.8） | 40.8→**60.8** | 7→**5** |
- GoldRecall 全改善。AllGoldHit は hotpot 大勝、musique +1、**2wiki のみ −0.4**（1000q で顕在、
  ただし coverDense の −1.5 から圧縮）。Precision@1 全改善＝トップのゴミ減。min-k 減＝読む量減。

### 6.6 Stage 1 × Stage 2 合成（trf, 300q）
| データセット | baseline-S1 + coverSoft | ResiRAG-S1 + coverSoft | ResiRAG-S1 + score |
|---|---|---|---|
| 2wiki | 67.0 / 36.3 | 63.1 / 33.0 | 56.6 / 34.0 |
| musique | 44.4 / 10.3 | 44.4 / **13.3** | 37.0 / **14.7** |
| hotpot | 68.5 / 42.7 | 67.2 / 40.3 | 52.7 / 29.0 |
- 普遍的相乗なし。musique（under-prop 支配）でのみ ResiRAG-S1 が AllGoldHit 改善。

### 6.7 JEMHopQA（日本語マルチホップ, ja_ginza, 120q, grouped gold）
`research/eval_jemhopqa.py`。evidence 三つ組から**ホップ単位の gold グループ**を構成:
値を含むチャンク（精密, 74%）、無ければ主語タイトルの全チャンク（記事レベル fallback, 26%）。
AllGoldHit = 全ホップ満たす。Contain と違い YES/NO 45 問も評価可能。

| 区分 | n | score R/AGH | coverSoft R/AGH |
|---|---|---|---|
| 全体 | 120 | 50.4 / 23.3 | 54.8 / **32.5**（AGH +9.2）|
| comparison | 73 | 52.7 / 23.3 | 63.4 / **43.8**（AGH +20.5）|
| compositional | 47 | 46.8 / 23.4 | 41.5 / **14.9**（AGH −8.5）|
| YES/NO | 45 | 51.1 / 24.4 | 57.8 / 40.0（+15.6）|

- **差別化達成**: 日本語でも coverSoft が効き、全体 AllGoldHit +9.2（英語より大きい）。移植が言語を跨ぐ。
- **手法レベルの知見**: coverSoft は **comparison 型（独立 2 記事）で大勝、compositional 型
  （ブリッジ連鎖）で悪化**。被覆＝多様性が「2 つの別エンティティを覆う」comparison に本質的に適合。
  全体は comparison の勝ちが compositional の負けを上回り純プラス。
- **符号不一致の切り分け（済）**: fallback を除外し**精密 gold（値-in-チャンク）のみ**で再測定しても
  符号は不変（comparison AGH +14.8、compositional AGH −6.4、全体 +4.9）。
  → **compositional 悪化は gold 定義のアーティファクトでなく本物の手法特性**（`--precise-only`）。
  英語 musique（compositional）では coverSoft が効いたのに日本語 compositional では逆効果 ＝
  **英日で compositional の符号が genuinely 異なる**。原因候補: ginza NER 品質、または手法×問題型の
  相互作用。今後掘る価値のある本物の謎（測定ノイズではない）。

### 6.8 headroom（trf, 300q, Stage 2 の天井）
| | 2wiki | musique | hotpot |
|---|---|---|---|
| AllGoldHit@5 現在 | 36.3 | 9.7 | 31.3 |
| top-20→5 再ランカー上限 | 49.0 | 23.3 | 55.3 |
| 冗長性（top-5 が上位と共有） | 83% | 71% | 87% |

---

## 7. 確定した知見

**肯定的**:
1. 失敗は二峰性、δ では両立不能（介入実験で裏付け）。
2. ボトルネックは Stage 1 でなく Stage 2。AllGoldHit が律速。
3. coverSoft（密アンカー＋活性化 IDF 飽和被覆）が GoldRecall@5 を全データセットで改善、
   文脈効率（Precision@1・min-k）も改善。トークンフリー・レイテンシ増ほぼゼロ。
4. **効くのは汎用の重複除去でなくグラフ活性化被覆**（対照 coverDense>coverUniform>coverAll で実証）。
5. 増分索引が軽い（関係なし・LLM なし・大域要約なし）。coverSoft は索引に何も足さない＝利点保存。

**否定的（これも貢献）**:
6. MMR（汎用多様化）はマルチホップに逆効果。
7. 統一原理（残差）は entity 粒度でのみ有効、passage 粒度では不成立（Stage 2 残差は失敗）。
8. Stage 1×Stage 2 は普遍的に相乗しない（データ依存、under-prop 支配データでのみ）。
9. AllGoldHit は hotpot 以外動きにくい（2wiki −0.4）＝手法の弱点。効果の一部は dense 由来。

---

## 8. 論文としての枠組みと自己評価（正直版）

**主語**: 「新しい選択アルゴリズムの発明」ではなく、
「**トークンフリー関係フリー GraphRAG には診断可能な Stage 2 ボトルネックがあり、グラフ自身の
クエリ活性化を飽和被覆信号として使うと、汎用の多様化が失敗・逆効果になる場面でそれが解ける**」。
**e2e は主指標にしない**（reader LLM を交絡）。gold Recall/AllGoldHit ＋ 文脈効率で閉じる。

**貢献**: (a) 二峰性診断と δ 限界、(b) gold-passage 評価基盤＋日本語マルチホップ、
(c) 活性化被覆 Stage 2（対照でグラフ信号の寄与を実証）、(d) 否定的知見（MMR/残差）、
(e) 増分索引の運用優位。

**正直な弱み**: 手法新規性が薄い（被覆リランキングは Lin-Bilmes 以来既知）、主指標 AllGoldHit が
動きにくい、効果の一部が dense 由来、SoTA に未達、合成が拮抗。
→ **トップ会議のメソッド論文としては現状弱い。Findings/ワークショップ/ショートが射程。**

**格を上げるレバー（効く順）**:
1. **JEMHopQA（日本語マルチホップ）を当てる** — 差別化の一枚看板。既存資産で回せる。
2. **coverSoft を LinearRAG 以外の候補にも載せる**（例: HippoRAG）→「汎用トークンフリー選択」に昇格。
   incremental 批判を一般性で封じる。最有効だが設計が重い。
3. 主語を GoldRecall＋効率に寄せる（AllGoldHit の弱さを弱点でなくする）。
4. e2e を 1 データセットだけ安全弁に。
5. 増分索引のマイクロベンチ（追加コスト vs LLM 系）。

**投稿前に潰す先行研究**: (a) 要約の劣モジュラ被覆（Lin-Bilmes 系）、(b) 多様化 PPR /
coverage-based selection、(c) SPRIG（arXiv:2602.23372, 同路線＝ NER 共起+PPR・CPU-only）。

**自己批判**: オーケストレータ（Claude）は「統一原理(残差)」という美しい物語に二度引っ張られ、
ResiRAG と Stage 2 残差(C) に時間を使ったが両方限定的だった。数字が強いのは地味な coverSoft。
分岐の多くはユーザーのツッコミ（dense が強いだけでは/MMR 下げてるだけでは/ランク固定は恣意的/
e2e よりゴミを測れ）が方向を正した。**「美しい話」より「勝った話」を推す**のが教訓。

---

## 9. コード資産の目録

すべて元 skill を非破壊で拡張。

| ファイル | 役割 | 主要関数 |
|---|---|---|
| `.claude/skills/linearrag/scripts/resirag.py` | Stage 1 改良＋ Stage 2 実験フック | `_residual_query`, `activate_entities_residual`, `ResiRetriever.__call__`（元一致）, `stage2_candidates`（内部量を返す） |
| `research/build_gold_datasets.py` | 正典コーパス→ gold_ids 付きデータ | `build(name)` |
| `research/eval_gold_recall.py` | gold-passage 評価器 | `evaluate`, strategy=ids/title/sentence |
| `research/rank_headroom.py` | Stage 2 伸び代探針 | 順位分布・冗長性・天井 |
| `research/stage2_experiment.py` | 選択戦略比較 | `select_score/cover/cover_soft/residual/mmr`, `--focus`, `--s1` |
| `research/recall_curve.py` | 文脈効率（LLM フリー） | Recall@k / Precision@k / min-k |

`stage2_candidates(query, top_n, **s1_kw)`: Stage 1+PPR を 1 回回し、top_n 候補の
`cand_rows/rel(PPR)/activation/levels/query_vec/cand_ids/cand_titles` を返す。`s1_kw` で
Stage 1 を baseline（既定）/ ResiRAG に切替。これで全選択戦略が同一 Stage 1 を共有し公平に比較できる。

**索引**: `linearrag_index/{2wiki,hotpot,musique}_hpr_trf`（en_core_web_trf）。
`_hpr`（sm 版）は旧。`jemhopqa`（ja_ginza）は日本語。

---

## 10. 現在地と次の手

**完了**: 失敗診断 / 評価基盤（正典・trf・gold_ids・検証） / baseline 確定 / ResiRAG（効果小） /
coverDense→coverSoft（新規性対照・full 1000 確定・文脈効率） / Stage 1×2 合成（データ依存）/
マスター文書。

**次（推奨順）**:
1. **JEMHopQA（日本語マルチホップ）を gold_ids 方式で評価**（レバー1・一枚看板）。
   `dataset/jemhopqa/` は既にある（evidence 三つ組・corpus.jsonl）。gold を ids 化して coverSoft を測る。
   注意: 現状 JEMHopQA の gold は title 戦略（evidence 主語）。ids 化 or title 戦略の精緻化が要る。
2. レバー2（coverSoft を他手法の候補に載せて汎用化）。
3. γ 最終決定・α/top_n 感度・full 1000 で coverSoft を各データセット再確認。
4. 増分索引マイクロベンチ、関連研究の最終サーベイ。

---

## 11. 再現コマンド集（WSL, `cd /root/work/LinearRAG-Skill`）

```bash
# 正典データ生成（/tmp に *_q.json,*_c.json を curl 済み前提）
python3 research/build_gold_datasets.py

# trf 索引構築（各 20〜35 分）
uv run .claude/skills/linearrag/scripts/build_graph.py \
  --input dataset/2wiki_hpr/corpus.jsonl --output linearrag_index/2wiki_hpr_trf \
  --lang en --spacy-model en_core_web_trf

# baseline gold-recall
uv run research/eval_gold_recall.py --index linearrag_index/2wiki_hpr_trf \
  --questions dataset/2wiki_hpr/questions.json --strategy ids

# 選択戦略比較（focus = A/B/C + 参照）
uv run research/stage2_experiment.py --index linearrag_index/musique_hpr_trf \
  --questions dataset/musique_hpr/questions.json --top-n 50 --limit 300 --focus \
  --gammas 0.5 --res-strengths 0.5 --s1 baseline   # --s1 resirag で合成

# 文脈効率
uv run research/recall_curve.py --index linearrag_index/hotpot_hpr_trf \
  --questions dataset/hotpot_hpr/questions.json --top-n 50 --gamma 0.5 --limit 400

# headroom
uv run research/rank_headroom.py --index linearrag_index/2wiki_hpr_trf \
  --questions dataset/2wiki_hpr/questions.json --limit 300
```

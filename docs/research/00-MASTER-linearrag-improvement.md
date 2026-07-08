2

# LinearRAG 改良研究 — マスターノート（生きた文書）

> **これが唯一の正典。以後はこのファイルを更新していく。**
> 冷えた状態で読んでも、やってきたこと・仕組み・数字・各判断の理由が再構成できることを目標に書く。
> 個別ノートは詳細の付録。数字と結論はここに集約する。
> 最終更新: 2026-07-05

---

## ★ 最重要アップデート（2026-07-05）: 研究の重心が「手法」から「診断」へ移った

2日目に外部研究と強ベースラインを当てて、手法路線が三重に閉じ、代わりに**新しい診断**that's the real prize が出た。§12 に詳細。要点だけ先に:

1. **SETR（EMNLP2025, LG AI）が「集合選択」の枠組みを先取り済み** — coverSoft の"Select段発明"は主張不可。SETR は LLM(8B)版。
2. **GPT-4o 集合選択（上限）に coverSoft は gain の ~30% しか届かない**。「トークンフリーで匹敵」は否定。
3. **22M クロスエンコーダ（60ms/CPU, API $0）が GPT-4o とほぼ同水準、coverSoft を明確に上回る** → coverSoft は
   コスト-品質フロンティア上で **dominated（劣位）**。手法としては終了。
4. **しかし——診断が反転を生んだ（Hidden Hop）**: マルチホップ失敗は「選択」でなく「**候補生成**」に集中する。
   クエリ不可視な 2 ホップ目 gold（隠れホップ）は BM25/dense プールに無く、**グラフ(PPR)だけが橋伝播で拾う**
   （プール被覆 +12〜27pt）。ただし注入だけでは回答に化けず（クエリ基準の選択にも不可視）、連鎖 CE でも半分止まり。
5. **グラフに唯一無二の役割が確定**: 検索器でも選択信号でもなく「**隠れホップ候補の注入器**」。SETR/リランカーが
   最適化する"選択"段は、そもそも間違った段だと実測で示す（逆張りだが数字で支持）。

→ **論文の形**: 「手法が勝つ」でなく「**The Hidden Hop: マルチホップ検索は選択でなく候補生成で失敗する**」という
   診断・分析論文。coverSoft は"読まない床"、クロスエンコーダは"安い読解"、グラフは"隠れ候補注入器"として梯子に整列。

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
| ------------ | --------- | ------------ | ------------ |
| 2wiki        | 53.5%     | 63.1%        | 35.2%        |
| hotpot       | 54.7%     | 51.8%        | 29.5%        |
| musique      | 39.6%     | 37.4%        | 11.0%        |

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

| 選択戦略                           | 2wiki                 | musique               | hotpot                |
| ---------------------------------- | --------------------- | --------------------- | --------------------- |
| score（元 LinearRAG）              | 63.5 / 36.3           | 37.8 / 9.7            | 53.5 / 31.3           |
| **coverSoft γ=0.5（採用）** | **67.0 / 36.3** | **44.8 / 10.3** | **68.8 / 43.0** |
| coverDense（飽和なし）             | 66.4 / 35.3           | 43.9 / 10.3           | 67.5 / 40.3           |
| dense（密のみ）                    | 54.0 / 22.7           | 37.1 / 7.7            | 65.3 / 38.3           |
| MMR                                | 56.0 / 24.7           | 31.9 / 2.3            | 46.2 / 21.7           |
| oracle（上限）                     | 80.8 / 59.0           | 69.7 / 40.7           | 88.0 / 76.3           |

### 6.3 新規性の対照実験（trf, 300q, GoldRecall、同じ密アンカー、被覆信号だけ変える）

| 被覆信号                                   | 2wiki          | musique        | hotpot         |
| ------------------------------------------ | -------------- | -------------- | -------------- |
| **coverDense（活性化×IDF）**        | **66.4** | **43.9** | **67.5** |
| coverUniform（活性化・均等重み）           | 63.0           | 40.0           | 53.8           |
| coverAll（全エンティティ・均等＝汎用除去） | 35.3           | 27.1           | 42.8           |

- 単調に coverDense > coverUniform > coverAll。汎用除去(coverAll)が最悪＝「any dedup works」を否定。
- 活性化への限定で激変（2wiki +27.7pt）、活性化重みでさらに上乗せ（hotpot +13.7pt）。
- 「ただ dense が強い」も否定: dense 単体は 2wiki/musique で baseline に負けるが coverDense は勝つ。

### 6.4 完成ペア修正 A/B/C（trf, 300q）

| 手法                          | 2wiki R/AGH           | musique R/AGH         | hotpot R/AGH          |
| ----------------------------- | --------------------- | --------------------- | --------------------- |
| **A: coverSoft γ=0.5** | **67.0 / 36.3** | **44.8 / 10.3** | **68.8 / 43.0** |
| B: coverDual                  | 65.7 / 34.7           | 41.9 / 9.3            | 62.8 / 38.3           |
| C: residual@0.3               | 56.3 / 23.7           | 37.0 / 6.0            | 66.8 / 39.3           |

- A 採用（2wiki AllGoldHit 退行が 300q で解消、hotpot AGH +2.7）。B 不採用。C 棄却（§5.4）。

### 6.5 coverSoft@0.5 の baseline 比（full 1000q, 確定）＋ 文脈効率（400q）

| データセット | GoldRecall@5                   | AllGoldHit@5                   | Precision@1          | min-k中央値      |
| ------------ | ------------------------------ | ------------------------------ | -------------------- | ---------------- |
| 2wiki        | 63.1 →**66.0**（+2.9）  | 35.2 → 34.8（−0.4）          | 74.5→**86.5** | 4→**3**   |
| musique      | 37.4 →**44.5**（+7.1）  | 11.0 →**12.0**（+1.0）  | 47.8→**57.2** | 17→**12** |
| hotpot       | 51.8 →**67.0**（+15.2） | 29.5 →**42.3**（+12.8） | 40.8→**60.8** | 7→**5**   |

- GoldRecall 全改善。AllGoldHit は hotpot 大勝、musique +1、**2wiki のみ −0.4**（1000q で顕在、
  ただし coverDense の −1.5 から圧縮）。Precision@1 全改善＝トップのゴミ減。min-k 減＝読む量減。

### 6.6 Stage 1 × Stage 2 合成（trf, 300q）

| データセット | baseline-S1 + coverSoft | ResiRAG-S1 + coverSoft | ResiRAG-S1 + score   |
| ------------ | ----------------------- | ---------------------- | -------------------- |
| 2wiki        | 67.0 / 36.3             | 63.1 / 33.0            | 56.6 / 34.0          |
| musique      | 44.4 / 10.3             | 44.4 /**13.3**   | 37.0 /**14.7** |
| hotpot       | 68.5 / 42.7             | 67.2 / 40.3            | 52.7 / 29.0          |

- 普遍的相乗なし。musique（under-prop 支配）でのみ ResiRAG-S1 が AllGoldHit 改善。

### 6.7 JEMHopQA（日本語マルチホップ, ja_ginza, 120q, grouped gold）

`research/eval_jemhopqa.py`。evidence 三つ組から**ホップ単位の gold グループ**を構成:
値を含むチャンク（精密, 74%）、無ければ主語タイトルの全チャンク（記事レベル fallback, 26%）。
AllGoldHit = 全ホップ満たす。Contain と違い YES/NO 45 問も評価可能。

| 区分          | n   | score R/AGH | coverSoft R/AGH                   |
| ------------- | --- | ----------- | --------------------------------- |
| 全体          | 120 | 50.4 / 23.3 | 54.8 /**32.5**（AGH +9.2）  |
| comparison    | 73  | 52.7 / 23.3 | 63.4 /**43.8**（AGH +20.5） |
| compositional | 47  | 46.8 / 23.4 | 41.5 /**14.9**（AGH −8.5） |
| YES/NO        | 45  | 51.1 / 24.4 | 57.8 / 40.0（+15.6）              |

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

### 6.8 BM25 基準線（同一正典コーパス・同一 gold 指標, full 1000q）— 必須の sanity check

`research/eval_bm25.py`（rank-bm25, LLM/NER/埋め込みなし）。GoldRecall@5 / AllGoldHit@5:

| データセット | BM25(text)  | BM25(title+text)      | LinearRAG(score) | coverSoft             |
| ------------ | ----------- | --------------------- | ---------------- | --------------------- |
| 2wiki        | 58.1 / 26.4 | 63.9 / 31.2           | 63.1 / 35.2      | **66.0 / 34.8** |
| hotpot       | 65.5 / 39.2 | **71.0 / 46.6** | 51.8 / 29.5      | 67.0 / 42.3           |
| musique      | 34.4 / 7.5  | 40.4 / 9.7            | 37.4 / 11.0      | **44.5 / 12.0** |

- **BM25 は極めて強い**。title+text で LinearRAG baseline(score) を GoldRecall 全データセットで上回る。
  「グラフが BM25 に勝つ」通説は単発検索・この指標では不成立。
- **hotpot は BM25(title+text) が coverSoft に勝つ**（71.0>67.0, 46.6>42.3）＝ hotpot は語彙で解ける
  「見かけのマルチホップ」。graph の優位が出ない。
- **coverSoft は真に難しい 2wiki・musique で BM25 に勝つ**（GoldRecall +2.1 / +4.1）。
- **フェアネス**: coverSoft は BM25(text) には全データセットで勝つ。BM25 にタイトルを与えると hotpot のみ逆転。
- **主張の修正**: 「全データセットで SoTA」でなく「**真のマルチホップ(2wiki/musique)で BM25 含む
  基準線を上回る。hotpot は語彙で解ける例外**」。より誠実で守れる。
- **含意**: BM25 は必須基準線（無しで投稿は即死だった）。**レバー2（coverSoft を BM25 候補に載せる）が
  最重要に格上げ** — BM25+coverSoft > BM25 なら「選択レイヤーは語彙検索にも効く」＝ hotpot の脅威を反転。
- **関連（要精読）**: "Shifting from Ranking to **Set Selection** for Retrieval"（ACL2025, aclanthology
  2025.acl-long.861）＝我々の「Select 段」の直接先行の可能性。差別化の要。

### 6.9 レバー2: coverSoft は BM25 候補にも効く（汎用選択レイヤー, full 1000q）

`research/eval_bm25_coversoft.py`。BM25(title+text) が候補 top-50 を生成 → coverSoft が
グラフ活性化被覆＋密アンカーで再選択（活性化は graph 由来＝トークンフリー維持）。

| データセット | BM25        | **BM25+coverSoft**             | LinearRAG+coverSoft |
| ------------ | ----------- | ------------------------------------ | ------------------- |
| 2wiki        | 63.9 / 31.2 | **67.4 / 36.1**（+3.5 / +4.9） | 66.0 / 34.8         |
| hotpot       | 71.0 / 46.6 | **73.6 / 51.2**（+2.6 / +4.6） | 67.0 / 42.3         |
| musique      | 40.4 / 9.7  | **45.1 / 12.8**（+4.7 / +3.1） | 44.5 / 12.0         |

- **coverSoft は BM25 候補を全データセット・両指標で改善**（GoldRecall +2.6〜4.7, AllGoldHit +3.1〜4.9）。
  → **選択レイヤーは検索器に依存しない汎用部品**（graph 候補にも lexical 候補にも効く）。
- **hotpot の BM25 脅威が反転**: 昨夜「BM25(71.0/46.6) > LinearRAG+coverSoft(67.0/42.3)」だったが、
  **BM25+coverSoft(73.6/51.2) が全体最良**。「BM25 に負ける」が「BM25 すら改善する」に。
- **グラフの価値が移動**: 候補生成でなく**選択信号**として活性化被覆が効く（候補が BM25 でも graph でも）。
- **論文の格上げ**: 「LinearRAG のパッチ」→「**任意の検索器の上に載る、トークンフリーな選択レイヤー
  (Select 段)**」。incremental 批判を一般性で封じる（レバー2の狙い達成）。

### 6.10 headroom（trf, 300q, Stage 2 の天井）

|                              | 2wiki | musique | hotpot |
| ---------------------------- | ----- | ------- | ------ |
| AllGoldHit@5 現在            | 36.3  | 9.7     | 31.3   |
| top-20→5 再ランカー上限     | 49.0  | 23.3    | 55.3   |
| 冗長性（top-5 が上位と共有） | 83%   | 71%     | 87%    |

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

**主語（レバー2 後にアップグレード）**: 「新しい選択アルゴリズムの発明」でも「LinearRAG の改良」でもなく、
「**RAG パイプラインに欠けている『Select 段』を、グラフのクエリ活性化を飽和被覆信号にした
トークンフリーな選択レイヤーとして与える。これは検索器に依存せず（LinearRAG でも BM25 でも）
候補集合を改善し、汎用の多様化(MMR)が逆効果になるマルチホップでこそ効く**」。
BM25 という強基準線すら上回る（BM25+coverSoft が全体最良）ことで「安いが弱い」批判を回避。
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

| ファイル                                        | 役割                               | 主要関数                                                                                                                         |
| ----------------------------------------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `.claude/skills/linearrag/scripts/resirag.py` | Stage 1 改良＋ Stage 2 実験フック  | `_residual_query`, `activate_entities_residual`, `ResiRetriever.__call__`（元一致）, `stage2_candidates`（内部量を返す） |
| `research/build_gold_datasets.py`             | 正典コーパス→ gold_ids 付きデータ | `build(name)`                                                                                                                  |
| `research/eval_gold_recall.py`                | gold-passage 評価器                | `evaluate`, strategy=ids/title/sentence                                                                                        |
| `research/rank_headroom.py`                   | Stage 2 伸び代探針                 | 順位分布・冗長性・天井                                                                                                           |
| `research/stage2_experiment.py`               | 選択戦略比較                       | `select_score/cover/cover_soft/residual/mmr`, `--focus`, `--s1`                                                            |
| `research/recall_curve.py`                    | 文脈効率（LLM フリー）             | Recall@k / Precision@k / min-k                                                                                                   |

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

---

## 12. 2日目（2026-07-04〜05）: SETR衝突 → クロスエンコーダ → Hidden Hop 診断

### 12.1 先行研究の衝突（SETR）

- **SETR** "Shifting from Ranking to Set Selection for RAG"（Lee et al., EMNLP2025, LG AI Research）。
  `docs/paper/2507.06838v2.pdf`。**我々の「Select段」枠組み・被覆目的・検索器非依存・同データセットを先取り**。
  中身は fine-tuned Llama-3.1-8B（GPT-4oで40K蒸留, 16×A100）＝ **LLM版**。推論時CoTで276〜409トークン/クエリ。
- 影響: 「集合選択の発明」は主張不可。残る差分は**トークンフリー**の一点のみ。要精読の隣人: GraphER(オンラインLLMフリー
  グラフ再ランキング, arXiv:2603.24925)。

### 12.2 LLM上限ベースライン（`research/eval_llm_upperbound.py`, GPT-4o, 同候補・同gold, 300q）

プロンプトはSETR Figure2逐語。raw（LLM選択そのまま）とpad（5枠補完）の両報告。API応答はキャッシュ。

| selector         | 2wiki     | hotpot    | musique   |
| ---------------- | --------- | --------- | --------- |
| bm25             | 65.7/35.3 | 70.8/46.0 | 38.4/8.0  |
| bm25+coverSoft   | 68.0/38.0 | 74.0/53.3 | 43.6/12.7 |
| bm25+gpt4o(raw)  | 71.7/44.7 | 85.5/73.7 | 50.7/17.0 |
| bm25+gpt4o(+pad) | 74.1/47.7 | 89.7/80.0 | 53.9/19.7 |

- **coverSoft は GPT-4o gain の ~20〜42%（GoldRecall）しか回収しない**。「無料で匹敵」は否定。
- GPT-4o は平均1.7〜2.1枚しか選ばない（SETRの2.5〜3.4と整合）。

### 12.3 クロスエンコーダ（`research/eval_crossencoder.py`, ms-marco-MiniLM-L-6-v2 22M, CPU 60ms, 300q）

| selector                | 2wiki               | hotpot              | musique             |
| ----------------------- | ------------------- | ------------------- | ------------------- |
| bm25+coverSoft          | 68.0/38.0           | 74.0/53.3           | 43.6/12.7           |
| **bm25+crossenc** | **71.5/43.7** | **84.8/70.7** | **49.6/17.7** |
| bm25+gpt4o(raw)         | 71.7/44.7           | 85.5/73.7           | 50.7/17.0           |

- **22M CE が GPT-4o とほぼ同水準**（GPT-4o gain のほぼ100%を回収）。musiqueはAGHでGPT-4o超え。
- **CE は coverSoft を全データで明確に上回る**（hotpot 70.7 vs 53.3）。
- → coverSoft は**パレート劣位**（精度も低く、Tri-Graph索引が要る分インフラも重い）。手法として終了。
- 上向きの示唆: pointwise 読解の22M CE が 8B LLM集合選択に並ぶ = **SETRの「大LLM要る」前提を疑う**。

### 12.4 Hidden Hop 診断（`research/eval_hiddenness.py`, 実験A+B, 300q）

hiddenness(gold) = 1 − クエリ類似度（BM25語彙+密の平均パーセンタイル）。質問ごとに最も隠れたgoldで層別。

**実験A: 隠れ度四分位別 AllGoldHit@5**（2wiki例）— 見える85〜100% → 深隠れ**全員0%**。GPT-4oもCEも同じく崩れる。
**実験B: all-gold が候補プールに入る率**:

|         | BM25プール | PPRプール | union | graph効果       |
| ------- | ---------- | --------- | ----- | --------------- |
| 2wiki   | 48%        | 59%       | 65%   | +17pt           |
| hotpot  | 87%        | 76%       | 94%   | +7pt            |
| musique | 27%        | 41%       | 54%   | **+27pt** |

隠れビンに集中: musique隠れビン BM25 4%→PPR 32%, 深隠れ BM25 0%→PPR 17%。
**→ ボトルネックは選択でなく候補生成。グラフ(PPR)だけが橋伝播で隠れホップをプールに届ける（＝グラフの唯一の役割）。**

### 12.5 融合プールは回答に化けない（`research/eval_fusion.py`, 実験C, 300q）

BM25∪PPR ラウンドロビン融合 top-50 → 選択。プール被覆は +12〜16pt 上がるが:

|         | プール被覆(bm25→fusion) | crossenc   | coverSoft  |
| ------- | ------------------------ | ---------- | ---------- |
| 2wiki   | 48→61                   | 43.7→44.0 | 38.0→36.7 |
| musique | 27→44                   | 17.7→20.0 | 12.7→11.0 |

- **注入した隠れ候補を pointwise 選択が拾えない**（+0〜2ptのみ, coverSoftはむしろ悪化）。
- 理由: 隠れホップはクエリに似ていない ＝ **検索でも選択でもクエリ基準に不可視（二重の不可視性）**。

### 12.6 連鎖条件付きCE（`research/eval_chain_ce.py`, 実験D, 300q, w=0.5）

枠1はクエリ類似CE、枠2以降は「クエリ⊕既選択passage本文」で候補を再採点する貪欲選択（LLM/訓練なし）。

|         | fusionプール被覆 | fusion+pointwise | **fusion+chain** |
| ------- | ---------------- | ---------------- | ---------------------- |
| 2wiki   | 60.7             | 44.0             | 46.3（+2.3）           |
| hotpot  | 91.0             | 70.7             | 71.0（+0.3）           |
| musique | 43.7             | 20.0             | 22.0（+2.0）           |

- **連鎖は融合プール上でのみ効く**（bm25プールでは効果ゼロ）＝診断と整合。だが **+2ptと小さい**。
- **プール被覆(60.7/43.7)と達成(46.3/22)の巨大ギャップ**が残る（14〜22pt未変換）。理由: 隠れホップは橋"エンティティ"を
  共有するだけで本文が似ておらず、テキストCEの連鎖でも弱くしか浮かない。**安い読解では根本的に破れない。**

### 12.7 現在地（2日目終了時点）

- **手法で勝つ路線は三重に閉じた**（SETR先取り / GPT-4o上限 / CE支配）。coverSoftは主役を降りる。
- **代わりに Hidden Hop 診断that IS the contribution**: マルチホップ失敗=候補生成の隠れホップ問題、二重の不可視性、
  グラフだけが橋伝播でプールを埋める、だが注入も連鎖も部分的にしか変換できない（＝問題の根深さの定量化）。
- **次**: (a) この診断を軸に分析論文へ統合（コスト梯子§12.2-3 + Hidden Hop §12.4-6）、(b) もう一段の悪あがき
  （強いCE=bge-reranker-v2-m3 で連鎖、エンティティ橋シグナルでの連鎖）、(c) 応用(Obsidian)へ。
- コード: `research/eval_{llm_upperbound,crossenc oder,hiddenness,fusion,chain_ce}.py`。GPT-4o選択キャッシュ
  `research/cache_llm_*.json`（再課金なしで再採点可）。

### 12.8 ★強い選択器フェーズ: グラフ注入 + GPT-4o が候補生成ボトルネックを突破（`research/eval_fusion_llm.py`, 150q）

制約（トークンフリー）を一旦外し、「グラフが注入した隠れホップを、強い set-aware 選択器なら拾えるか」を測定。
同じ BM25∪PPR 融合プール上で、安い CE（MiniLM）と強い GPT-4o を対決。BM25プール選択は既存キャッシュ再利用。

|         | 融合プール天井 | bm25+gpt4o | **fusion+gpt4o**  | fusion+crossenc |
| ------- | -------------- | ---------- | ----------------------- | --------------- |
| 2wiki   | 57.3           | 42.0       | **54.0**（+12.0） | 39.3（+0.6）    |
| hotpot  | 92.7           | 76.0       | 78.7（+2.7）            | 66.0（0）       |
| musique | 43.3           | 16.0       | **28.0**（+12.0） | 18.7（+3.4）    |

（AllGoldHit@5。musique は当初 API 失敗25件で 21.3% と過小 → billing 修正後 fails=0 で 28.0%）

- **プール注入gainの変換率: GPT-4o 62〜90% vs クロスエンコーダ 0〜18%**。**同じ融合プールで、強い選択器だけが隠れホップを拾う**。
- 2wiki は fusion+gpt4o 54.0 がプール天井 57.3 の **~94%** を回収。musique は +12pt（変換率62%）。hotpot は元々易しく伸び小。
- **確定した弧**: マルチホップの壁=候補生成の隠れホップ → **グラフがトークンフリーに注入**（BM25/CE/GPT-4o単独には原理的に不可）
  → 安い選択器は拾えない → **強い set-aware 選択器（橋を渡る読解）だけが変換**。各部品の役割が完全に確定。
- **含意（低コスト化の射程）**: LLM が本当に要るのは「融合プール上の**隠れホップ橋選択**」という一点に局所化された。
  SETR（全パイプラインに8B LLM）と違い、この狭いスキルだけ蒸留・近似すればよい。教師データ = GPT-4o の融合プール選択
  （`research/cache_llm_fusion_*.json` に蓄積中）。→ 次フェーズ: この一点を安く近似できるか。

### 12.9 運用ルール（PC安定性）

重いクロスエンコーダ/trf を回すと WSL が全4コアを飽和し **Windowsホストがフリーズ**（3回発生、OOMではない）。
**対策**: 重い CPU ジョブは必ず `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false`
で2コアに制限（load 0.7〜1.4で安定）。軽い MiniLM 単独は無制限でも可。GPT-4o はネットワークなので CPU 無負荷。

### 12.10 ★注入器 shoot-out: graphPPR が被覆でも回答でも最接近ライバルに勝つ（`research/eval_injectors.py`, 150q）

ユーザーの仮説「グラフ注入が勝ち筋」を直接検証。同じ BM25 top30 ベースに、各注入器が +20候補足す。
全て query-time 安価（行列積、LLM/訓練なし）。dualEntity ＝ BridgeRAG の候補拡張部品（entity-ANN）。

**プール被覆（all-gold in pool）と回答（GPT-4o-mini選択, AllGoldHit@5）**:

|                          | 2wiki 被覆/回答       | hotpot 被覆/回答      | musique 被覆/回答     |
| ------------------------ | --------------------- | --------------------- | --------------------- |
| bm25                     | 42.7 / 39.3           | 85.3 / 73.3           | 18.7 / 12.0           |
| +dense                   | 44.0 / 39.3           | 91.3 / 77.3           | 34.7 / 16.7           |
| +dualEntity(BridgeRAG流) | 43.3 / 40.7           | 86.0 / 76.0           | 25.3 / 14.0           |
| **+graphPPR**      | **56.7 / 51.3** | **92.7 / 79.3** | **44.0 / 22.7** |

**隠れ度別プール被覆（隠れホップの領域）**:

- 2wiki 隠れビン: bm25/dense/dualEntity 全て 8% vs **graphPPR 41%**（+33pt）。深隠れ: 全0 vs graphPPR 5%（唯一）。
- musique 隠れビン: 3/19/14% vs **graphPPR 38%**。深隠れ: 0/8/0% vs **graphPPR 13%**。

**結論**:

1. **隠れホップの領域は graphPPR の独壇場**（多段橋伝播 vs 1ジャンプの dense/ANN）。深隠れの尾に届く唯一の注入器。
2. **被覆優位が回答優位に化ける**: 対dense で 2wiki +12 / musique +6（回答AllGoldHit）。hotpot は易しく僅差(+2)。
3. **dense/dualEntity は回答をほぼ上げない**（2wiki dense=bm25=39.3）＝届かないから足しても無意味。
4. **BridgeRAG の注入部品(dualEntity)を被覆・回答の両方で上回った**＝「ANN拡張より多段グラフPPRが深い隠れホップに届く」実測。

**論文の骨格（確定）— 主張A（トークンフリー注入）を本体に**:

- 幹: 「隠れホップ問題を hiddenness で層別定量化し、注入器を直接対決させ、**トークンフリーな多段グラフPPR注入が
  BridgeRAG流ANN拡張/dense近傍の届かない深い隠れホップに届く唯一の安い注入器**」であることを被覆＋回答で示す。
- 先行を正面から引く: HippoRAG(PPR多ホップ、ただしLLM構築)、BridgeRAG(注入+LLM judge)、StratRAG(bridge難を観察)。
  差分 = 診断の解像度(hiddenness分解) ＋ 注入のトークンフリー性 ＋ ANN拡張への実測勝利。
- 主張B（選択もトークンフリー）は未解決（安い選択4連敗）。上限のみ提示（fusion+GPT-4o §12.8）＝将来課題。
- 出口: YANS(7/31, 診断で出せる) → NLP2027 / 国際 Findings。

### 12.11 RM3/PRF 注入器を追加（主張の calibration, 300q プール被覆）
論文の「唯一の安価注入器」最上級を検証するため、古典の疑似適合フィードバック RM3（BM25上位10件から
拡張語20、元クエリ3倍重み＋拡張語で再検索、上位を注入）を shoot-out に追加（eval_injectors.py --rm3）。

全体プール被覆: 2wiki bm25 47 / dense 50.3 / dualEntity 47.7 / **rm3 53.3** / graphPPR 59.7。
musique bm25 21 / dense 36.7 / dualEntity 27 / rm3 31.7 / graphPPR 43.0。hotpot rm3 88.0 / graphPPR 90.7。

隠れ度別（隠れ / 深隠れ）:
- 2wiki: rm3 21/4, graphPPR 43/4（深隠れは **rm3 が graphPPR と同点4%**）
- musique: rm3 7/0, dense 23/7, graphPPR 33/**12**（深隠れ graphPPR のみ有意, dense 7, rm3 0）
- hotpot 隠れ: rm3 57, dense 61, graphPPR 68

**含意（主張のトーンダウン）**: RM3 は本物の競合。2wiki では dense 超えの2位、隠れ・深隠れで graphPPR に
肉薄〜同点。「テキスト拡張は届かない」は 2wiki では言えない。**ただし RM3 は不安定**（musique 隠れで7%＝
bm25 並みに崩れる）。dense も musique で強いが 2wiki で0。**一貫して隠れホップに届くのは graphPPR だけ、
最難の musique 深隠れ(12%)で唯一有意にリード**。
→ 論文の主張: 「唯一(only)」→「**最も頑健で一貫した(most robust/consistent)**」に修正。最強の古典
ベースライン(RM3)を土俵に載せた上での calibrated な勝ち。過大主張を防いだ（RM3を測らずに投稿していたら
2wiki 同点を突かれていた）。

### 12.12 ★確定版: 注入器 shoot-out プール被覆 full-1000（LLMなし・無料）
`eval_injectors.py --limit 1000`。300q(§12.10-11)より頑健。**graphPPR が全データセット・全隠れビンで単独最良**
（300q で唯一あった 2wiki 深隠れ rm3 同点も解消）。

全体プール被覆: 2wiki 46.0/dense51.0/dual47.7/rm3 53.0/**graphPPR 59.2**。
hotpot 83.1/89.4/84.9/88.0/**89.9**。musique 25.2/41.3/30.1/33.9/**45.6**。

隠れ度別（隠れ / 深隠れ, all-gold in pool）:
| | bm25 | dense | dualEntity | rm3 | **graphPPR** |
|---|---|---|---|---|---|
| 2wiki | 11/0 | 18/4 | 13/0 | 24/3 | **40/6** |
| musique | 3/0 | 18/6 | 7/1 | 9/0 | **26/12** |
| hotpot(隠れ) | 44 | 59 | 49 | 55 | **63** |

- **graphPPR は全ビンで単独最良**、差は最難・最深で最大（**musique 深隠れ 12% vs dense 6%＝2倍**、他≤1%）。
- 最強の安価競合は **dense**（rm3 は 2wiki 強・musique 崩れで不安定）。
- 最終主張（calibrated）: 「唯一」でなく「**全データセット・全隠れ度で全ての安価注入器を上回り、深いほど差が拡大**」。

---

## 13. ★SPRIG head-to-head → 枠組みの再構築（2026-07-06）
研究の骨格が一段深くなった日。eval_sprig_h2h.py で「同じトークンフリーNER+PPRグラフを、SPRIGは
"検索器"、我々は"注入器"として使う」直接対決を組んだら、想定外の発見が連鎖した。

### 13.1 発見1: 純エンティティ種PPRは"検索器"として実は最強（前提崩壊）
sprigGraph（クエリ固有名詞だけを種にPPR、passage種=0、DPR混合なし、PPRスコアで直接ランキング）が、
2wiki で AllGoldHit@5=57.7%（300q）／プール被覆@50=80%。**我々のLinearRAG baseline(35.3)も、
GPT-4o融合選択(54.0)も超えた**。「グラフは検索器として弱い→だから注入器」という§12の前提が崩れた。

### 13.2 バグ検証（/tmp/sprig_debug.py）— 本物と確定
- **A. passage種PPRは構造上注入不能**: top-50が種と**100%一致**。二部グラフで段落種は(1-d)r項が毎回
  元段落を押し上げ、新段落へは W²（d²=0.25）でしか届かず直接項に負ける。→ sprigHybrid/sprigRRF が
  bm25/rrf と同値なのはバグでなく数学的必然。SPRIG論文が GraphRRF>Graph としたのと逆の理由。
- **B. エンティティ種PPRの深隠れヒットは本物の多段伝播**: BM25で2403位のNicki Minaj頁をPPR3位に。
  種=クエリ固有名詞のみ、gold情報なし。経路（種→hop1頁→橋エンティティ→hop2頁）を6ケース目視確認。
- 結論: LinearRAG baseline(35.3) < 純エンティティPPR(57.7) の差は、本家 eq.7 の**DPR混合がクエリ類似へ
  引っ張り隠れホップを埋めていた**ため。本家は自分の伝播力を殺していた。

### 13.3 発見2: 棲み分けの法則（regime complementarity, full-1000 プール被覆@50）
隠れ度別で、クエリ基準と純エンティティPPRが**鏡像**。（可視ビン / 深隠れビン）
| | BM25 | 純entPPR | 融合fuseRRF |
|---|---|---|---|
| 2wiki | 100 / 0 | 88 / 60 | 100 / 60 |
| hotpot | 100 / 48 | 75 / 68 | 100 / 78 |
| musique | 90 / 0 | 62 / 29 | 99 / 32 |
- クエリ基準=可視の王・隠れで全滅／純entPPR=隠れの王・可視で崩れる。**hiddenness軸が勝者を予言**。

### 13.4 統合: 融合プール（可視BM25/RRF ∪ 隠れ純entPPR）が全データセット最良
全体プール被覆@50（full-1000）: 2wiki fuseRRF 83.2（BM25 47.9 / entPPR 76.6）、
hotpot 94.6（86.0 / 72.8）、musique 58.7（29.6 / 40.6）。**旧ours(BM25+LinearRAG-PPR)を全面更新**
（DPR混合PPRでなく純entPPRを注入源にすべき）。

### 13.5 選択のトークンフリー化: スロット選択（主張Bに前進）
4連敗の死因＝全選択器が「クエリ類似」で採点→隠れホップ沈む。純entPPRスコアは「橋への近さ」でクエリ類似と
独立→隠れホップが浮く。設計は**混ぜるな枠で分けろ**: top-5 を可視枠(RRF上位)＋隠れ枠(entPPR上位)にスロット。
AllGoldHit@5（full-1000, 安価勢は無料 / 4oは融合150q）:
| | RRF | entPPR | rrf3way混合 | **slot** | GPT-4o |
|---|---|---|---|---|---|
| 2wiki | 33.3 | 55.6 | 44.1 | **57.1** | 54.0 |
| hotpot | 54.0 | 40.0 | 62.0 | **62.5** | 78.7 |
| musique | 13.0 | 17.7 | 18.3 | **21.7** | 28.0 |
- **slot が安価勢を全データセットで上回る**（混合rrf3wayも超え＝「枠分け>混合」実証）。
- **2wiki で無料slot(57.1) > GPT-4o(54.0)**＝トークンフリー選択が4o超えの初例。
- hotpot/musique はまだ4oに負け（読解勝負のhotpotは特に-16.2）。**主張Bは部分解決**。

### 13.6 未解決の恣意性（次の一手）
slot の分割 slot32(可視3隠れ2) vs slot23(2/3) は**データで最適が割れる**（2wiki=slot23, hotpot=slot32）。
「2/3」が恣意的（ユーザー指摘）。→ 次: **team-draft interleaving（RRF1位→entPPR1位→…重複スキップ、
パラメータフリー・引用可）** で恣意性を消せるか。効けば「一致度で自動適応」（可視質問は自然にRRF深掘り）。

### 13.7 論文への含意
物語が「診断→注入器」から「**診断→棲み分け法則→種設計の自滅を特定→融合＋スロット選択**」に進化。
選択(主張B)が2wikiで解けたことで「診断だけ」から「解決まで届く」手前に。ビジュアル: docs/research/02-VISUAL-hidden-hop.html。

### 13.8 恣意性の除去: パラメータフリー選択（team-draft / min-rank, full-1000）
slot の分割比 2/3 が恣意的（ユーザー指摘, データ間で最適が割れる: 2wiki=slot23, hotpot=slot32）。
2つのパラメータフリー選択を追加:
- **teamdraft**: RRF↔entPPR の公平な交互ドラフト（重複スキップ、開始順のみ）。
- **min-rank**: 各候補を「2リストの**良い方の順位**で入場」（RRFは順位を"和"で取るので隠れホップが埋まる；
  min は片方1位なら1位で入場）。対称・分割比なし・開始順なし＝完全パラメータフリー。

AllGoldHit@5（full-1000）:
| | rrf3way和 | teamdraft | **minrank** | 調整slot最良 | GPT-4o(150q) |
|---|---|---|---|---|---|
| 2wiki | 44.1 | 50.8 | **52.6** | 57.1 | 54.0 |
| hotpot | 62.0 | 62.4 | **61.5** | 62.5 | 78.7 |
| musique | 18.3 | 20.9 | **21.7** | 21.7 | 28.0 |

**結論**:
1. **min-rank は原理的で堅い**: 順位和の混合(rrf3way)を 2wiki/musique で明確に超え(+8.5/+3.4)、hotpotは同点。
   調整版slotに hotpot/musique で並び、2wiki のみ -4.5。**恣意的ノブなしで混合に勝ち調整版に肉薄**。
2. **ただし恣意性除去のコスト**: 調整slot23(57.1)が誇った「2wikiでGPT-4o(54.0)超え」は、パラメータフリー
   min-rank では 52.6 と**4o のわずか下**に落ちる。→ 誠実な言い回しは「**トークンフリー選択が GPT-4o に匹敵
   (2wiki 52.6 vs 54.0, 誤差内)、しかも無料**」。§推奨A の "beat→match" 修正が数字で裏付いた。
3. **混合はプールを良くするが選択を悪くする**（musique: rrf3way プール61.9>fuse58.7 なのに AllGoldHit
   18.3<minrank21.7）＝「混合は深いプールに隠れホップを入れても top-5 で埋める」の綺麗な実例。

### 13.9 ★現行LLM(gpt-5.4-mini)参照 → 主張の再定位「候補生成こそ主役」（2wiki 300q）
「GPT-4oは古い(2024)、現行LLMでは？」（ユーザー, 推奨B）。gpt-5.4-mini（miniの最新, gpt-5.5-mini は不在）を
**同一 fuseRRF プール**で選択させ、トークンフリー選択と一騎打ち。さらに**同じLLMをBM25プールでも**回して
候補生成の価値を分離。eval_sprig_h2h.py --llm --llm-pool {fuseRRF,bm25} --model gpt-5.4-mini。

AllGoldHit@5（2wiki 300q, 同一subset）:
| 選択器 × プール | 値 |
|---|---|
| gpt-5.4-mini × BM25プール | 43.0 |
| **min-rank(無料) × 融合プール** | 54.3 |
| slot23(調整) × 融合プール | 60.7 |
| **gpt-5.4-mini × 融合プール** | **77.7** |

3つの読み（いずれも型番に強い）:
1. **候補生成の価値（選択器固定）**: gpt-5.4-mini が BM25 43.0 → 融合 77.7（**+34.7**）。選択器不変、
   プールだけで+35。**「ボトルネックは選択でなく候補生成」を現行LLMで実証**。
2. **完全無料 > 現行LLM+素朴検索**: 融合+min-rank(54.3, $0) > gpt-5.4-mini+BM25(43.0)（+11.3）。
3. **選択自体はLLMが上（正直）**: 同一融合プールで LLM 77.7 > min-rank 54.3（+23.4）。良い候補あれば読解はLLM。

**主張の最終再定位**: 「トークンフリー選択がLLMに勝つ」（脆い・型番依存）を捨て、「**マルチホップの真の
ボトルネックは候補生成。それをトークンフリーに解く融合プールが、最強クラスのLLM選択器を+35押し上げ、
完全無料パイプラインですら素朴候補のLLMに勝つ**」へ。**新しいLLMほど良いプールを欲しがる＝時代に強い**。
注意: gpt-4o(§6.1 旧BM25∪PPRプールで54.0)との単純比較は不可（プール構成が別）。
残: hotpot/musique の pool-value 測定（任意, 予算次第）、full-1000 化。

### 13.10 ホップ深さ層別（MuSiQue 2/3/4-hop, full-1000）— スケーラビリティに機序で回答
「もっと難しいタスクでも使える？最大何ホップ届く？」（ユーザー）。MuSiQue の type フィールド
（2hop/3hop*/4hop*, gold数=ホップ数）で層別。eval_hopdepth.py。

| hop | n | mean_hidden | bm25 | rrf | entPPR | fusion | 最難goldのPPR順位(中央) | gold∈PPR50 |
|---|---|---|---|---|---|---|---|---|
| 2 | 518 | 0.102 | 44 | 57 | 58 | 76 | **16** | 68% |
| 3 | 316 | 0.169 | 19 | 38 | 28 | 56 | **268** | 58% |
| 4 | 166 | 0.218 | 4 | 7 | 8 | 10 | **1893** | 47% |

3予言すべて確認:
1. **hiddenness の崖はホップ数で鋭化**: mean_hidden 0.102→0.169→0.218 単調増（深いホップ=よりクエリ不可視）。
2. **★エンティティPPR到達の d^k 減衰を可視化**: 最難gold の PPR 順位中央値 **16→268→1893**（1ホップごと
   ~16×,~7× の幾何級数）。d^k（1ホップで質量減衰）が順位の幾何後退として現れた。2hopは余裕でtop-50、
   4hopはコーパス下位15%へ押し出し。
3. **4hopで全手法が床に崩落**（fusion 10）: 橋の鎖が長すぎ単一パスPPRの伝播が枯渇。

**含意（Limitations節に昇格）**: 手法は2〜3hopで有効・fusionが全深さで最良、**4hopで壁**（到達の幾何枯渇が機序）。
診断は深さで鋭くなるが単一パスPPR到達が追いつかない。→ 将来: **反復再シード**（hop-k結果から再PPR,
HippoRAG-2的な反復検索のトークンフリー版）で壁を越えられるか。「スケールするか？」に減衰の数式込みで正直回答。

### 13.11 pool-value を全データセットに拡張（gpt-5.4-mini, 300q）— 隠れ度に比例
§13.9 を hotpot/musique にも（eval_sprig_h2h.py --llm-pool both で一発、両プール選択）。

AllGoldHit@5:
| | LLM@BM25 | LLM@融合 | **Δ(プール価値)** | 無料min-rank@融合 |
|---|---|---|---|---|
| 2wiki(隠れ多) | 43.0 | 77.7 | **+34.7** | 54.3 |
| musique(最難) | 16.3 | 29.0 | **+12.7** | 22.7 |
| hotpot(可視多) | 76.0 | 82.7 | **+6.7** | 61.7 |

3つとも診断が予言する並び:
1. **プール価値 ∝ 隠れ度**: 2wiki +34.7 > musique +12.7 > hotpot +6.7。hiddennessが高いデータほど
   トークンフリー候補生成が同じLLMを大きく押し上げる。「候補生成こそ主役」が全データで裏付き。
2. **完全無料 > 現行LLM+素朴検索（隠れ多で成立）**: 2wiki 54.3>43.0, musique 22.7>16.3。
   hotpot は可視多で BM25候補が既に良く 61.7<76.0（正直に記載）。
3. **選択はLLMが上（一貫）**: 同一融合プールで LLM > min-rank（+23.4/+21.0/+6.3）。
キャッシュ: research/cache_llmref_{ds}_gpt-5.4-mini_{bm25,fuseRRF}_300.json。

---

## 14. 競合再検証と指標合わせ — 立ち位置の最終確定（2026-07-08）

### 14.1 競合の再検証（ネット, 「本当に論文価値あるか」）
悪い発見2つ:
- **Regime-Conditional Retrieval (arXiv:2604.09019, 2026-04)**: 2ホップ検索を Q-dominant（hop-2エンティティが
  クエリに出る=可視）/ B-dominant（橋パッセージのみ=隠れ）に分ける。**「棲み分けの二分法」は先取りされた**。
  ただし二値述語・学習ルーター(2wiki訓練)・両腕ともテキスト側(+5pp程度)。連続指標/グラフ伝播/プール分析なし。
- **Calibrated Fusion (arXiv:2603.28886, 2026-03)**: PPR×ベクトルのパーセンタイル較正スコア融合(LastHop@5
  +1.4〜2.2pp)。**「グラフ+語彙の融合」も既出**（SPRIG GraphRRFに続き2本目）。なぜ効くかの分析ゼロ。
- 白状: min-rank ≈ 古典 CombMAX（Fox & Shaw 1994, 要引用）。「passage種は注入不能」は damping=0.5 条件付き
  （SPRIGはα=0.15=伝播強め）→数式で条件明示に弱める。
- セーフ確認: Weakest Link (2601.12499)は読解側の位置バイアス＝別層。
- **残る独自資産**: ①連続hiddenness計器＋崖(GPT-4o深隠れ0%) ②種設計の機序(reset質量+100%重複+DPR自滅)
  ③pool-value×隠れ度(+34.7/+12.7/+6.7) ④d^k減衰の可視化(16→268→1893)。
- 判定: 「発見の論文」→「**計器と機序の論文**」に再定位。分野がこの空き地に月単位で収束中＝**急いで書く**
  （YANS 〆切7/31）。

### 14.2 ★指標合わせ（eval_metrics_align.py, full-1000）— システムとしても公表フロンティア級
ユーザー「精度でも良い線では？」→ 正しかった。競合の物差し(GoldRecall@k, LastHop@5)で再測定:

| GoldR@5 | HippoRAG(LLM-KG構築) | **うちmin-rank($0)** | slot23 |
|---|---|---|---|
| musique | 52.1 | **52.3 同等** | 51.2 |
| hotpot | 76.2 | **79.5 上回る** | 77.6 |
| 2wiki | 89.5 | 78.3 負け | 80.8 |

- **HippoRAG正典コーパス使用＝この比較はガチ同一土俵**。トークンフリーが LLM構築KG に 2/3 で同等以上。
  2wiki のみ負け＝LLM関係抽出の価値が残る領域（正直に分析）。
- **vs SPRIG** (R@10, 2wiki): うち **87.0** vs GraphRRF 79.4（+7.6, トークンフリー同士）。
- **vs Calibrated Fusion** (LastHop@5, 2wiki): **ベースライン錨が一致**（うちBM25 51.4 ≈ 彼ら51.7）。
  そこから彼ら+2.2(→53.6)、**うち+25.0(→76.4, しかも前ホップ条件なしの難設定)**。
  musique の LastHop は設定が別物（彼らベースライン75 vs うち24）＝比較対象外と明記。
- 論文の最終形: **「分析（計器+機序+pool-value）＋その診断が導いた無料システムが公表フロンティア級」の両輪**。

# 論文執筆マスター: The Hidden Hop

> **これは論文を書くための整理ノート**（章立て＝論文構成順）。時系列の実験ログは
> `00-MASTER-linearrag-improvement.md`。数字はすべてそちらの実験に基づく。
> 作成 2026-07-05。想定投稿: YANS2026（ポスター, 申込〜7/31）→ NLP2027 / 国際 Findings。

---

## 0. 一段落サマリー（アブストラクトの種）

マルチホップ検索の失敗は「選択（reranking）」でなく「**候補生成**」で起きる。2ホップ目の正解
パッセージはしばしば**クエリと語彙的にも意味的にも似ておらず**（＝*hidden hop*）、BM25・dense・
クロスエンコーダ・LLM のような**クエリ基準の手法には検索でも選択でも不可視**である。我々は
*hiddenness* 指標で失敗を層別し、ボトルネックが候補生成にあることを定量化する。そして
**関係フリーのグラフ PPR による候補注入だけが、トークンフリーに hidden hop をプールへ届けられる**
ことを、dense-NN や BridgeRAG 型 entity-ANN との直接対決で示す（プール被覆・最終回答の両方で勝利、
特に深い hidden hop で独占的）。ただし注入された hidden hop を選び切るには強い set-aware 読解
（GPT-4o）が要り、安価な選択器は届かない——この「選択のトークンフリー化」を未解決課題として、
測定した上限とともに提示する。

**主張A（実証済み・本体）**: トークンフリーなグラフ PPR 注入は、全データセット・全隠れ度で
あらゆる安価注入器（dense-NN / RM3 テキスト拡張 / BridgeRAG 型 entity-ANN）を hidden hop 被覆で
上回り、hidden が深いほど差が拡大する最も頑健な注入器（full-1000: musique 深隠れ 12% vs 競合最良 6%）。
**主張B（未解決・提示のみ）**: 注入された hidden hop の選択をトークンフリーに近似する。

---

## 1. 問題設定と動機（Introduction）

### 1.1 マルチホップ検索の構造
質問「Xの母はいつ死んだ？」は 2 つの根拠を要する:
- **hop-1（可視ホップ）**: 「X の母は Y」——X が明示的に出るのでクエリと重なる。取りやすい。
- **hop-2（hidden hop）**: 「Y は851年に死去」——**クエリに X も「母」も出てこない**。橋エンティティ Y を
  介してのみ hop-1 と繋がる。クエリとの語彙・意味の重なりが低い。

### 1.2 主指標: AllGoldHit@k
マルチホップは「全ての根拠が揃って初めて答えられる」。よって主指標は
**AllGoldHit@k =「全 gold パッセージが top-k に揃った質問の割合」**（全か無か）。部分点つきの
GoldRecall@k は補助。gold はデータセット注釈のパッセージ ID（1 質問 2〜4 個）。

### 1.3 我々の問い
「マルチホップ検索の失敗はどこで起きているのか。選択の問題か、候補生成の問題か。
そして安価（トークンフリー）にどこまで解けるのか。」

---

## 2. 関連研究と立ち位置（Related Work）

| 研究 | 何をした | 我々との関係 / 差分 |
|---|---|---|
| **HippoRAG** (NeurIPS2024) | KG＋PPR で単一ステップ多ホップ検索。2wiki で dense比 +20% R@5 | **PPR で多ホップを跨ぐ**思想の源流。ただし KG 構築に **LLM(OpenIE)** ＝トークン課金。我々はトークンフリー |
| **SETR** (EMNLP2025) | reranking を **set selection** に再定式化、fine-tuned 8B LLM で被覆選択 | 「集合選択」の枠組みを先取り。**LLM版**。我々は「選択が正しくても候補に無ければ無意味（候補生成が壁）」を示し、論点を上流へ移す |
| **BridgeRAG** (2026) | training-free。**dual-entity ANN で hop-2 候補を拡張**＋bridge条件付き **LLM judge** で選択。「後段は元クエリでなく橋に条件付けて順位づけよ」 | **最接近**。注入部品(entity-ANN)を我々は shoot-out で被覆・回答とも上回る。選択に LLM judge が要る点も差 |
| **SPRIG** (arXiv:2602.23372) | NER共起グラフ＋PPR、CPU-only・token-free。BM25/dense を **seed** に。GraphRRF が 2wiki R@10 0.794 (RRF 0.697) | **同じ土俵（トークンフリーGraphRAG）の隣人**。だが**グラフを検索器(seed→PPRランキング)として使う**。**"なぜ効くか"の分解（hop/hidden 層別）が無い**。Limitation に「LinearRAG/HippoRAG との head-to-head 無し」「future work: musique」と明記＝我々の空き地 |
| **StratRAG** (2026, ベンチ) | bridge 質問が最難、「橋エンティティはクエリに現れない」と観察 | **現象の観察のみ**。我々は機序を診断＋解決 |
| **Beam Retrieval** (NAACL2024) | エンコーダをホップ横断で end-to-end 訓練。2wiki 精度99.9% | **訓練型の王者**。ただし **gold-hop 注釈つき訓練が必須**＝注釈の無い実コーパス/日本語で使えない。我々はトークンフリー・訓練フリー |

**我々の隙間**: 「トークンフリーGraphRAGが効く**現象**は既知（SPRIG/HippoRAG）。だが**なぜ効くか＝
hidden hop**を層別で診断し、**注入器として**（検索器でなく）位置づけ、最接近の安価注入器
（dense-NN / BridgeRAG entity-ANN）に**直接勝つ**ことを示した研究は無い。」

---

## 3. 評価基盤（Setup / Reproducibility）

- **コーパス**: HippoRAG `reproduce/dataset` の正典（PropRAG/HippoRAG2 と同一検索プール）。
  `dataset/{2wiki,hotpot,musique}_hpr`。段落＝パッセージ、`gold_ids` をパッセージ単位で保持
  （musique の同一タイトル別段落は (title,text) で厳密同定）。規模 6,119 / 9,811 / 11,656。
- **索引**: 関係フリー Tri-Graph（LinearRAG, arXiv:2510.10114）を **en_core_web_trf** で構築
  （sm→trf で GoldRecall +12.3pt、NER 品質が支配的）。埋め込み paraphrase-multilingual-mpnet-base-v2。
- **検証済み**: 索引 id == corpus 行、gold の本文一致 100%、id 経路の傍証（recall@50≠0）。
- **元 LinearRAG baseline（trf, full 1000, AllGoldHit@5）**: 2wiki 35.2 / hotpot 29.5 / musique 11.0。
- **hiddenness 定義**: `hidden(gold) = 1 − ½(dense類似度パーセンタイル + BM25語彙パーセンタイル)`。
  質問ごとに**最も隠れた gold**で層別（AllGold の律速だから）。

---

## 4. 診断: 失敗は候補生成の hidden hop に集中する（Analysis 1）

### 4.1 hiddenness で層別すると全手法が同じ崖で死ぬ（300q, AllGoldHit@5, 2wiki 例）
| 隠れ度四分位 | bm25 | coverSoft | crossenc(22M) | GPT-4o |
|---|---|---|---|---|
| 可視 | 85% | 97% | 100% | 91% |
| やや隠れ | 47% | 51% | 64% | 75% |
| 隠れ | 9% | 4% | 11% | 15% |
| **深く隠れ** | **0%** | **0%** | **0%** | **0%** |

→ **深い hidden hop では GPT-4o すら 0%**。選択器の強さの問題ではない。

### 4.2 理由はプール（候補生成）: hidden hop はそもそも候補に無い（300q, all-gold in pool）
| ビン | BM25プール | +graphPPRプール |
|---|---|---|
| 隠れ (2wiki) | 15% | **44%** |
| 深隠れ (2wiki) | 0% | 5% |
全体プール被覆: 2wiki BM25 48%→union 65%、musique 27%→54%、hotpot 87%→94%。

→ **ボトルネックは選択でなく候補生成**。GPT-4o の失敗も「候補に無いものは選べない」で説明できる。

### 4.3 注入だけでは回答に化けない（二重の不可視性, 300q）
融合プール（BM25∪PPR）はプール被覆を +12〜16pt 上げるが、**pointwise 選択は変換できない**
（fusion+crossenc: 2wiki +0.3, musique +3.4）。**hidden hop は検索でも選択でもクエリ基準に不可視**。

---

## 5. 手法/発見: グラフ PPR は hidden hop に届く唯一の安価注入器（Analysis 2 / Method）

### 5.1 役割の再定義（本研究の鍵）
LinearRAG のグラフを**検索器（自分で top-k を決める）でなく、注入器（候補プールに hidden hop を足す）
として使う**。検索器としては BM25 に劣る（例: hotpot AllGoldHit 元LinearRAG 29.5 vs BM25 46.0）が、
注入器としては唯一無二。橋エンティティ経由の**多段 PPR 伝播**が、クエリ類似度が届かない hop-2 に届く。

### 5.2 注入器 shoot-out（150q）— 同じ BM25 top30 ベースに各注入器が +20 候補
全て query-time 安価（行列積、LLM/訓練なし）。dualEntity = BridgeRAG の候補拡張部品。

**プール被覆 / 最終回答（GPT-4o-mini 選択, AllGoldHit@5）**:
| 注入器 | 2wiki 被覆/回答 | hotpot 被覆/回答 | musique 被覆/回答 |
|---|---|---|---|
| bm25 | 42.7 / 39.3 | 85.3 / 73.3 | 18.7 / 12.0 |
| +dense | 44.0 / 39.3 | 91.3 / 77.3 | 34.7 / 16.7 |
| +dualEntity (BridgeRAG流) | 43.3 / 40.7 | 86.0 / 76.0 | 25.3 / 14.0 |
| **+graphPPR (ours)** | **56.7 / 51.3** | **92.7 / 79.3** | **44.0 / 22.7** |

**隠れ度別プール被覆（full-1000, all-gold in pool, 隠れ/深隠れ）**:
| 注入器 | 2wiki | musique | hotpot(隠れのみ) |
|---|---|---|---|
| bm25 | 12 / 0 | 3 / 0 | 49 |
| +dense | 15 / 0 | 23 / 7 | 61 |
| +dualEntity (BridgeRAG流) | 12 / 0 | 9 / 0 | 53 |
| +rm3 (RM3/PRF, 古典テキスト拡張) | 21 / 4 | 7 / 0 | 57 |
| **+graphPPR (ours)** | **43 / 4** | **33 / 12** | **68** |

### 5.3 読み取り（主張A, calibrated, full-1000 で確定）
1. **graphPPR が全データセット・全隠れビンで単独最良**（300q で唯一あった 2wiki 深隠れ rm3 同点も
   full-1000 で解消: graphPPR 6% > rm3 3%）。全体被覆 2wiki 59.2 / musique 45.6 / hotpot 89.9。
2. **差は最難・最深で最大**: **musique 深隠れ graphPPR 12% vs 競合最良 dense 6%（2倍）**、rm3/dualEntity ≤1%。
   musique 隠れ 26 vs dense 18 vs rm3 9。2wiki 隠れ 40 vs rm3 24 vs dense 18。
3. **被覆優位が回答優位に化ける**（150q, mini）: 対dense で 2wiki +12 / musique +6（回答 AllGoldHit）。
4. **最強の安価競合は dense**（rm3 でなく）: rm3 は 2wiki で強いが musique 隠れで崩れ（9%）、不安定。
   dense は 2 番手で安定だが graphPPR に全ビンで負ける。
5. **BridgeRAG の注入部品(dualEntity)には全データセット・被覆/回答で勝利**。
6. hotpot は hidden が浅く僅差(graphPPR 63 vs dense 59)＝**hidden が深いデータ(2wiki/musique)ほど差が開く**、診断と整合。

**主張の正確な言い回し（最上級「唯一/only」は使わない）**: dense・RM3 も**一部の hidden hop には届く**
（例: dense は musique 深隠れ 6%）。正しくは「**多段グラフ伝播は、全データセット・全隠れ度で全ての安価注入器
（dense-NN / RM3 テキスト拡張 / BridgeRAG entity-ANN）を hidden-hop 被覆で上回り、hidden が深いほど差が拡大する
最も頑健な安価注入器である**」。RM3 を土俵に載せた上での calibrated な勝ち（過大主張の回避）。

---

## 6. 上限と否定的知見: 選択のトークンフリー化は未解決（Analysis 3）

### 6.1 強い set-aware 選択器なら注入を変換できる（上限, 150q, fusion=BM25∪PPR プール）
| | fusionプール天井 | bm25+gpt4o | **fusion+gpt4o** | fusion+crossenc |
|---|---|---|---|---|
| 2wiki | 57.3 | 42.0 | **54.0** | 39.3 |
| hotpot | 92.7 | 76.0 | 78.7 | 66.0 |
| musique | 43.3 | 16.0 | **28.0** | 18.7 |
→ GPT-4o は注入 gain の 62〜90% を変換（2wiki 天井の94%）。**強い読解が hidden hop を橋渡しできる**。

### 6.2 安価な選択は hidden hop を選べない（4連敗、否定的知見）
| 試み | 結果 | なぜ失敗 |
|---|---|---|
| coverSoft（活性化被覆） | baseline 同点〜微減 | 見逃し gold の47%は未被覆活性エンティティ0 |
| pointwise CE (MiniLM/bge) | 融合プールを +0.6/+3.4 しか変換 | クエリ基準採点＝hidden hop が低スコア |
| 連鎖 CE（テキストで橋渡し） | +2pt（300q） | hidden hop は橋"エンティティ"共有で本文が非類似 |
| graph-bridge（グラフで橋渡し） | 36.7→33.3（悪化） | 橋エンティティを多 distractor が共有＝識別不能 |

→ **hidden hop の選択には意味的な橋渡し読解が要り、安価な構造/テキスト信号では代替できない**
（＝主張B が未解決である"理由"の定量化。これ自体が貢献）。

---

## 7. 貢献（Contributions）
1. **Hidden Hop 診断**: hiddenness 層別で、マルチホップ失敗が候補生成の hidden hop に集中し、
   検索・選択の**両段でクエリ基準に不可視**であることを定量化（GPT-4o すら深隠れで0%）。
2. **注入器としてのグラフの位置づけと実測勝利**: 関係フリーグラフ PPR は**トークンフリーに** hidden hop を
   プール注入する唯一の安価注入器であり、dense-NN・BridgeRAG型 entity-ANN を**被覆・回答の両方で上回る**
   （深隠れビンで独占）。SPRIG 等が観測した「グラフが効く」現象の機序を与える。
3. **選択の上限と否定的知見**: 注入は強い set-aware 読解でのみ変換可（上限を測定）。安価な選択 4 手法が
   なぜ失敗するかを示し、「選択のトークンフリー化」を明確な未解決課題として定式化。

---

## 8. 限界と未解決（Limitations / Future Work）
- **主張B（選択のトークンフリー化）が未解決**。上限（fusion+GPT-4o）のみ提示。教師データ
  （GPT-4o の融合プール選択, `research/cache_llm_fusion_*.json`）は将来の蒸留に利用可。
- 回答レベルの絶対値は **GPT-4o-mini** で測定（注入器の相対比較用）。**full 1000 + GPT-4o での確定が必要**。
- **RM3/PRF（古典のテキスト query expansion 注入器）を対照に未追加** ← 「唯一の安価注入器」の最上級を
  打つ前に必須。テキスト橋渡しの古典が hidden hop に届かないことを示せば主張が堅くなる。
- **SPRIG との head-to-head 未実施**（彼らの積み残しでもある）。seed vs inject の違いを実験で明示したい。
- 日本語（JEMHopQA）での hidden hop 再現は有望（分かち書き無し＝語彙一致が弱く hidden がより深刻なはず）。
- hotpot は hidden hop が浅く差が出にくい（正直に記載）。

---

## 9. 次にやるべき実験（To finish the paper）
優先順:
1. **RM3/PRF 注入器**を shoot-out に追加（テキスト拡張 vs グラフ伝播）。主張の生死を決める。
2. **full 1000q + GPT-4o** で 5.2 と 6.1 の確定数字（現在150q・mini）。予算: ~$30。
3. **SPRIG head-to-head**（GraphHybrid/GraphRRF を再現 or 近似し、hiddenness 層別を彼らの土俵で回す）。
4. **JEMHopQA** で診断＋注入 shoot-out を再現（差別化＋日本語の隠れホップ）。
5. 図: hiddenness × 各手法 AllGoldHit の崖グラフ、注入器別プール被覆の隠れ度曲線。

---

## 10. コードとデータ（Artifacts）
- `research/eval_gold_recall.py` 評価器 / `research/build_gold_datasets.py` 正典データ変換
- `research/eval_hiddenness.py` 診断（A/B、hiddenness 層別）
- `research/eval_fusion.py` / `eval_fusion_llm.py` 融合プールと GPT-4o 上限
- `research/eval_injectors.py` **注入器 shoot-out（被覆＋GPT-4o回答）** ← 本体実験
- `research/eval_crossencoder.py` / `eval_chain_ce.py` / `eval_graph_bridge.py` 安価選択の否定的知見
- `research/cache_llm_*.json` GPT-4o 選択キャッシュ（再課金なしで再採点／将来の蒸留教師）
- 索引 `linearrag_index/*_hpr_trf`（trf）。運用: 重い CPU ジョブは `OMP_NUM_THREADS=2` 必須（ホストフリーズ回避）。

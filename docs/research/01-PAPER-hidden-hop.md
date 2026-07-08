# 論文執筆マスター: The Hidden Hop

> **これは論文を書くための整理ノート**（章立て＝論文構成順）。時系列の実験ログは
> `00-MASTER-linearrag-improvement.md`。数字はすべてそちらの実験に基づく。
> 作成 2026-07-05。想定投稿: YANS2026（ポスター, 申込〜7/31）→ NLP2027 / 国際 Findings。

---

## 0. 一段落サマリー（アブストラクトの種）

（2026-07-08 λ訂正後の確定版。旧数字・「自滅」系の主張は§15参照＝撤回済み）

マルチホップ検索の失敗は「選択（reranking）」でなく「**候補生成**」で起きる。2ホップ目の正解
パッセージはしばしば**クエリと語彙的にも意味的にも似ておらず**（＝*hidden hop*）、BM25・dense・
クロスエンコーダ・LLM のような**クエリ基準の手法には検索でも選択でも不可視**である（深隠れは
GPT-4o 選択でも 0%）。我々は *hiddenness* 指標（連続値）で失敗を層別し、ボトルネックが候補生成に
あることを定量化する。機序: **グラフの多段伝播（文書→固有名詞→文書）だけが、クエリ類似度で並べる
手法には届かない hidden hop に橋を渡って到達する**（graphPPR 注入は dense/RM3/entity-ANN を圧倒:
2wiki 深隠れ 51% vs 次点 4%, full-1000, λ=0.05）。※旧「passage 種は注入不能(100%一致)/種が何を運ぶか」
は測り方のアーティファクトで撤回（§15.6）。seed-type は本質でない。
クエリ基準と エンティティ伝播は hiddenness 軸で**鏡像の棲み分け**をなし、正しく調整した LinearRAG
すら単一レジーム（可視偏重 hotpot で BM25 以下の 50.9）。処方は**直交レジームの min-rank 融合**
（CombMAX 流, パラメータフリー）: **トークンフリー枠の最良（忠実 LinearRAG）を全データで上回る**
（GoldR@5 +6.1/+28.6/+11.7）。さらに候補生成の価値を現行 LLM で定量: 同一 gpt-5.4-mini の候補だけ
差し替えて **+34.7/+12.7/+6.7（隠れ度に比例）**。
※「LLM 不使用で HippoRAG 級」は**見出しにしない**: それ自体は LinearRAG が主張済みで、現行の
HippoRAG 2 (R@5 90.4/96.3/74.7) には全データ大差負け＝ギャップは LLM 構築グラフの価値として正直に報告。

**主張A（実証済み・本体）**: hidden hop はクエリ基準手法に構造的に不可視で、エンティティ伝播を
運ぶ注入だけが届く（graphPPR は全安価注入器を全データ・全隠れ度で上回り、深いほど差が拡大:
2wiki 深隠れ 51% vs 次点 4%）。
**主張B（実証済み・システム）**: 単一レジームの強い手法は必ずどこかのレジームで崩れる。直交レジームの
min-rank 融合だけが全域で安定し、トークンフリー枠の最前線を全データで前進させる。
**主張C（未解決・提示のみ）**: 良い候補があっても選択は LLM が上（+21〜23）＝選択のトークンフリー化。

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

## 7. 貢献（Contributions）※ 2026-07-06 SPRIG h2h で大幅更新（§6.5参照）
1. **Hidden Hop 診断**: hiddenness 層別で、マルチホップ失敗が候補生成の hidden hop に集中し、
   検索・選択の**両段でクエリ基準に不可視**であることを定量化（GPT-4o すら深隠れで0%）。
2. **棲み分けの法則（regime complementarity）**: クエリ基準手法（BM25/RRF）と純エンティティ種 PPR は
   hiddenness 軸で**鏡像**——前者は可視を支配し隠れで全滅、後者は隠れの王で可視で崩れる。
   **hiddenness 軸が勝者を予言**する（full-1000 全データセット）。
3. **グラフ多段伝播の注入優位**（※旧「自滅」「passage種は注入不能」はいずれも撤回・§15.2/§15.6）:
   グラフの多段伝播だけが hidden hop に橋を渡って届き、**dense/RM3/entity-ANN を全データ・全隠れ度で圧倒**
   （2wiki 深隠れ 51% vs 次点 4%）。seed-type（passage か entity か）は本質でない。
   純エンティティ種は単独最強ではないが、**RRF との直交性が最も高く融合相方として最良**
   （minrank=RRF∪pureEnt > fuseLin=RRF∪linrag05, 全データ, 経験的）。
4. **融合システム + 候補価値の定量**: 直交レジームの min-rank 融合（CombMAX 流, パラメータフリー）が
   **トークンフリー枠の最良（忠実 LinearRAG λ0.05）を全データで上回る**（GoldR@5 +6.1/+28.6/+11.7、
   SPRIG にも 2wiki R@10 87.0 vs 79.4）。さらに **pool-value**: 同一 gpt-5.4-mini の候補だけ差し替えて
   +34.7/+12.7/+6.7（隠れ度に比例）＝候補生成ボトルネックの直接実証、LLM 世代に頑健。
   ※旧 HippoRAG(2024) との比較は参考行に留める（LinearRAG 自身が「トークンゼロで HippoRAG 級」を主張済み。
   現行 HippoRAG 2 には R@5 90.4/96.3/74.7 vs 78.3/79.5/52.3 で大差負け＝Limitations で正直に報告）。

---

## 8. 限界と未解決（Limitations / Future Work）
- **選択のトークンフリー化（主張C）は未解決**: 良い候補があっても、同一プール上で LLM 選択が
  min-rank を +21〜23 上回る（gpt-5.4-mini, 300q）。「LLM に勝つ」は現行モデルでは不成立（正直に）。
- min-rank は CombMAX (Fox & Shaw 1994) の再発見＝手法新規性なし。新規なのは「棲み分けた 2 リストでは
  SUM(RRF) でなく MAX を使うべき」の理由（順位和は hidden hop を埋める、実測 +8.5）。要引用。
- 「BM25 文書種は注入不能(100%一致)」は **d=0.5 での結果**（減衰率に条件付き、数式で明示する）。
- 日本語（JEMHopQA）での hidden hop 再現は有望（分かち書き無し＝語彙一致が弱く hidden がより深刻なはず）。
- hotpot は hidden hop が浅く候補生成の差は小さい（可視多）が、選択勝負では逆に差が出る（正直に記載）。
- ✅ **解決済み**: RM3/PRF 対照（§12.11-12, graphPPR が最頑健）、SPRIG head-to-head（§6.5, §13）、
  full-1000 プール被覆。

---

## 9. 次にやるべき実験（To finish the paper）
優先順:
1. **team-draft interleaving**（パラメータフリー選択）で slot の「2/3 恣意性」を消す。効けば「一致度で自動適応」。
2. **hotpot/musique の選択を 4o に迫る**別機構（橋条件の読解近似）。主張 B の完成度を上げる高リスク項。
3. **GPT-4o 比較を full-1000 に揃える**（現状 slot=1000q vs 4o=150q）。予算 ~$30。
4. **JEMHopQA** で診断＋棲み分け＋融合を再現（差別化＋日本語の隠れホップ）。
5. 図: ✅ ビジュアルガイド作成済み（docs/research/02-VISUAL-hidden-hop.html）。論文用に鏡像グラフ・崖グラフを清書。

---

## 10. コードとデータ（Artifacts）
- `research/eval_gold_recall.py` 評価器 / `research/build_gold_datasets.py` 正典データ変換
- `research/eval_hiddenness.py` 診断（A/B、hiddenness 層別）
- `research/eval_fusion.py` / `eval_fusion_llm.py` 融合プールと GPT-4o 上限
- `research/eval_injectors.py` **注入器 shoot-out（被覆＋GPT-4o回答）** ← 本体実験
- `research/eval_crossencoder.py` / `eval_chain_ce.py` / `eval_graph_bridge.py` 安価選択の否定的知見
- `research/cache_llm_*.json` GPT-4o 選択キャッシュ（再課金なしで再採点／将来の蒸留教師）
- 索引 `linearrag_index/*_hpr_trf`（trf）。運用: 重い CPU ジョブは `OMP_NUM_THREADS=2` 必須（ホストフリーズ回避）。

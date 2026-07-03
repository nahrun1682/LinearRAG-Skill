# 評価基盤（gold-passage recall、正典コーパス）

2026-07-03 ・ `research/build_gold_datasets.py` + `research/eval_gold_recall.py`

## 何を整備したか

英語データセットを **literatureと同じ検索プール**で、**パッセージ単位のgold** を持つ形に再構築した。
これで Contain@5 の表層ノイズに依存せず、機構が「根拠パッセージを実際に引けたか」を測れる。

- **出典**: OSU-NLP-Group/HippoRAG `reproduce/dataset/`（HippoRAG2/PropRAG/BridgeRAGと同じ正典）
- **再構築** (`build_gold_datasets.py`): `dataset/{2wiki,hotpot,musique}_hpr/` に
  - `corpus.jsonl`（title付き、id=コーパス通番。段落＝パッセージでチャンク分割なし＝gold対応が完全保存）
  - `questions.json`（`gold_ids` をパッセージ単位で保持。**gold欠損0件**）
  - musiqueは同一タイトルに複数根拠段落があるため title でなく (title,text) で厳密同定
  - `answer_aliases` も保持（Contain@5の偽陰性を低減）
- **索引**: 正しくケース保持されているので trf 不要、**en_core_web_sm で高速**（各2〜3分）
- **指標** (`eval_gold_recall.py`, strategy=ids):
  - Contain@5（answer∪aliasesの表層一致、旧指標・参考）
  - GoldRecall@5（各質問の gold のうち top-5 被覆率の平均）
  - **AllGoldHit@5**（全 gold が top-5 に揃った割合＝マルチホップを最後まで辿れたか、主指標）

| データセット | corpus | q | gold/q平均 |
|---|---|---|---|
| 2wiki_hpr | 6,119 | 1000 | 2.47 |
| hotpot_hpr | 9,811 | 1000 | 2.00 |
| musique_hpr | 11,656 | 1000 | 2.65 |
| jemhopqa（既存, title戦略） | 3,578 | 120 | — |

## 較正: baseline は忠実（delta=0.8が最良）

2wiki baseline の delta スイープ（300q, GoldRecall）: 0.8→51.2%, 0.5→47.9%, 0.3→46.3%, 0.1→44.7%。
**delta=0.8 が最良**で、旧コーパス用チューニングがこの正典コーパスでも有効。基盤は miscalibration でない。

## 盤石な基準値（full 1000, delta=0.8, sm NER）

| データセット | 条件 | Contain@5 | GoldRecall@5 | AllGoldHit@5 |
|---|---|---|---|---|
| 2wiki | baseline | 44.5% | 51.3% | 26.9% |
| 2wiki | ResiRAG-fuse | 45.0% | 51.2% | **27.4%** |
| hotpot | baseline | 53.1% | 49.1% | 25.3% |
| hotpot | ResiRAG-fuse | 53.0% | 48.6% | 24.5% |
| musique | baseline | 38.5% | 34.2% | 8.8% |
| musique | ResiRAG-fuse | 38.9% | 34.6% | **10.5%** |

## 所見

1. **旧chunks.json基準（2wiki 72.9% contain）は使えない**: 小さく易しい非正典コーパス＋タイトル欠落。
   今後は `_hpr` 正典コーパスを基準にする。
2. **LinearRAG(relation-free, sm)の素の到達率は重量級SoTAより低い**（2wiki GoldRecall 51%）。
   PropRAG(LLM命題+beam) 等との差は大きく、**改善の余地（headroom）が大きい**ことを意味する。
3. **ResiRAG-fuseの純増は正典でも小さい**（2wiki/musiqueで AllGoldHit +0.5〜+1.7pt、hotpotは微減）。
   musiqueで最大（+1.7pt AllGoldHit）＝under-propが多い難データで相対的に効く、という以前の傾向と整合。
4. **AllGoldHitが律速**: GoldRecall(片方は取れる) と AllGoldHit(両方揃う) の差が「マルチホップの壁」。
   例: 2wiki 51%→27%、musique 34%→9%。ここを上げるのが本丸（Stage 2再ランキング）。

## 未確認の較正変数（要判断）

- **sm vs trf NER**: sm は高速だがNERが弱く、グラフが疎になっている可能性。trf索引で baseline が
  大きく上がるなら、基盤のNERを trf に統一すべき。3データセットのtrf再構築は各15〜20分（OOM注意）。
- **naive dense recall@5** との対比（グラフの寄与量の定量化）。

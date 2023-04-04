# graduation_competition_3
## 愛媛大学 工学部 人工知能研究室 自然言語処理グループ

## 卒論コンペティション第三回(~2023.04/04)


- **TASK**
  - 日本語のTwitterテキストについての感情極性分類
  - 書き手の感情極性を５クラス分類(-2,-1,0,1,2)
  - 評価指標 Quadratic Weighted Kappa
- **DATASET**
  - Train/Dev/Test = 30k/2.5k/2.5k
  - 分割を変更してはならない
- 予定
  - 第一回 : ニューラルネットワークを使用しない
  - 第二回 : 外部データ(事前学習)を使用しない
  - 第三回 : モデル制約なし⭕️

## GPU制約
- 1人が同時に使えるGPUは1枚まで
- コンペに使えるのはTITAN RTXを搭載したサーバのみ（lepin, dyquem, luce, opus1, opus2, opus3）
- cuda :10.2（lepin, dyquem）
- cuda :11.3（opus[1-3], luce）

# 解法
## 概要
- [日本語言語理解ベンチマーク（JGLUE）](https://zenn.dev/hellorusk/articles/8e73cd5fb8f58e)を参考にLLMを採用した。
- 多言語モデルに対して、入力文を「日本語」「日本語[SEP]英語（翻訳文）」の２種類を実行した。
  - 翻訳文は、[JParaCrawl v3.0（big）](https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/)で日英翻訳したもの。
- 実験の結果より、「早稲田RoBERTa」と「京大DeBERTa」、「LUKE-large(lite)」によるアンサンブルを行った。


|表情筋が衰えてきてる。まずいな…[SEP]The facial muscles are waning.It's bad...|
|-|

|すき焼きに卵美味しそう〜[SEP]Sukiyaki and Eggs Look Delicious~|
|-|

## 感情分析による評価実験

### 大規模言語モデルの評価実験
1.validに対するQWKの表

F1はmacro平均によるもの。すべての値は%表記。

| models  | QWK | F1  |
| ------------- | ------------- | ------------- |
|cl-tohoku/bert-base-japanese-v2  |52.2|41.1|
|cl-tohoku/bert-large-japanese|56.4|41.6|
|||
|studio-ousia/luke-japanese-large|**63.6**|**47.0**|
|studio-ousia/luke-japanese-large-lite|59.2|40.3|
|megagonlabs/electra-base-japanese-discriminator|53.4|42.4|
|||
|nlp-waseda/roberta-large-japanese|63.0|45.5|
|ku-nlp/deberta-v2-base-japanese|59.1|43.9|
|ku-nlp/deberta-v2-large-japanese|62.8|46.0|

| multi-models  | QWK | F1  |
| ------------- | ------------- | ------------- |
|xlm-roberta-large|59.9|46.0|
|xlm-roberta-large（ja-en）|59.1|44.4|
|studio-ousia/mluke-large|59.4|44.1|
|studio-ousia/mluke-large（ja-en）|58.4|43.1|
|studio-ousia/mluke-large-lite|58.7|44.2|
|studio-ousia/mluke-large-lite（ja-en）|57.7|42.7|


2.testに対するQWKの表
| models  | QWK |
| ------------- | ------------- |
|cl-tohoku/bert-base-japanese-v2  |52.4|
|cl-tohoku/bert-large-japanese|55.6|
||
|studio-ousia/luke-japanese-large|**65.9**|
|studio-ousia/luke-japanese-large-lite|65.1|
|megagonlabs/electra-base-japanese-discriminator|57.6|
||
|nlp-waseda/roberta-large-japanese|64.3|
|ku-nlp/deberta-v2-base-japanese|62.9|
|ku-nlp/deberta-v2-large-japanese|64.9|

| multi-models  | QWK |
| ------------- | ------------- |
|xlm-roberta-large|59.0|
|xlm-roberta-large（ja-en）|**59.9**|
|studio-ousia/mluke-large|58.4|
|studio-ousia/mluke-large（ja-en）|**59.8**|
|studio-ousia/mluke-large-lite|58.8|
|studio-ousia/mluke-large-lite（ja-en）|**59.3**|

実験の結果、入力文は単言語よりも２言語を入力するほうが、性能が向上することを確認できた。
また、パラメータ数が多いlargeモデルの精度が高いことがわかる。

以下のデータより、「早稲田RoBERTa」と「京大DeBERTa」、「LUKE-large(lite)」によるアンサンブルを行った。

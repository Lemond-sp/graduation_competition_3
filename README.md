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
## BERTによる感情極性分類タスク
- 表を載せる

1.validに対するQWKの表
8モデルに対する評価
F1はmacro平均によるもの

| base-models  | QWK | F1  |
| ------------- | ------------- | ------------- |
|cl-tohoku/bert-base-japanese-v2  | wmt20(en)  | wmt20(en)  |
|cl-tohoku/bert-large-japanese|   |  |
|studio-ousia/luke-japanese-large|   |  |
|megagonlabs/electra-base-japanese-discriminator|   |  |
|nlp-waseda/roberta-large-japanese|   |  |
|ku-nlp/deberta-v2-base-japanese|   |  |
|ku-nlp/deberta-v2-large-japanese|   |  |
|xlm-roberta-large|   |  |

2.testに対するQWKの表
## 工夫点

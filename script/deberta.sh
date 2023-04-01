#!/bin/sh
# --gradient_accumulation_steps:2,3,4
# "--model_name" default="studio-ousia/luke-japanese-large-lite"
# "--model_name_short" default="large-lite"
# "--epochs", type=int, default=30
# exp01 : jumanpp なし
# exp02 : jumanpp あり（screen -r deb）
# cl-tohoku/bert-base-japanese-v2
# 'ku-nlp/deberta-v2-base-japanese'
python src/base.py \
      --model_name  ku-nlp/deberta-v2-base-japanese\
      --model_name_short deberta \
      --prepro_dir /home/kajikawa_r/competition/gradcomp/prepro/jumanpp \
      --sub_file /home/kajikawa_r/competition/gradcomp/ch03/submission/deberta/exp02.txt
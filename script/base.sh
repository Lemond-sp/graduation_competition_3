#!/bin/sh
# --gradient_accumulation_steps:2,3,4
# "--model_name" default="studio-ousia/luke-japanese-large-lite"
# "--model_name_short" default="large-lite"
# "--epochs", type=int, default=30

# cl-tohoku/bert-base-japanese-v2
# 'ku-nlp/deberta-v2-base-japanese'
python src/base.py \
      --sub_file /home/kajikawa_r/competition/gradcomp/ch03/submission/l-lite/exp03.txt
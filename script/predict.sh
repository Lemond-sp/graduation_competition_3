# cl-tohoku/bert-base-japanese-v2
# 'ku-nlp/deberta-v2-base-japanese'
python src/predict.py \
      --model_name  xlm-roberta-large\
      --model_name_short large-xlm \
     --model_path /home/kajikawa_r/competition/gradcomp/ch03/model/large-xlm/checkpoint-3752/pytorch_model.bin\
      --sub_file /home/kajikawa_r/competition/gradcomp/ch03/submission/large-xlm/exp03.txt
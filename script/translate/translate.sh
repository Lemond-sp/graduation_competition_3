REF_FILE='/home/kajikawa_r/competition/gradcomp/prepro/spm/valid.ja'
! fairseq-interactive /home/kajikawa_r/competition/gradcomp/ch03/jaen --path /home/kajikawa_r/competition/gradcomp/ch03/model/big/big.pretrain.pt --input $REF_FILE --batch-size 128 \
--remove-bpe sentencepiece \
--buffer-size 1024 --nbest 1 --max-len-b 50 \
--beam 5 \
| grep "^H-" | sort -V | cut -f3 > text.dev.txt
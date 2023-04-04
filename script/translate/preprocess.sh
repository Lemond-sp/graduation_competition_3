ENDICT='/home/kajikawa_r/competition/gradcomp/ch03/model/big/dict.en.txt'
JADICT='/home/kajikawa_r/competition/gradcomp/ch03/model/big/dict.ja.txt'
! fairseq-preprocess --source-lang ja --target-lang en --trainpref ../prepro/spm/train --validpref ../prepro/spm/valid --testpref ../prepro/spm/test \
                     --destdir jaen --srcdict $JADICT --tgtdict $ENDICT
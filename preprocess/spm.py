import os
import sentencepiece as spm
MODEL_PATH = "/home/kajikawa_r/competition/gradcomp/ch03/model/enja_spm_models/spm.ja.nopretok.model"
DATA_PATH = "/home/kajikawa_r/competition/gradcomp/data"
PREP_PATH = "/home/kajikawa_r/competition/gradcomp/prepro/spm"
sp = spm.SentencePieceProcessor()
sp.Load(MODEL_PATH)

# train
fout = open(os.path.join(PREP_PATH,"train.ja"), "w")
fin = open(os.path.join(DATA_PATH,"text.train.txt"), "r")
for line in fin:
    fout.write(" ".join(sp.EncodeAsPieces(line)) + "\n")
fin.close()
fout.close()
#! cp "/content/drive/MyDrive/translation/en.train.txt" "train.en"

# valid
fout = open(os.path.join(PREP_PATH,"valid.ja"), "w")
fin = open(os.path.join(DATA_PATH,"text.dev.txt"), "r")
for line in fin:
    fout.write(" ".join(sp.EncodeAsPieces(line)) + "\n")
fin.close()
fout.close()
#! cp "/content/drive/MyDrive/translation/en.valid.txt" "valid.en"

# test
fout = open(os.path.join(PREP_PATH,"test.ja"), "w")
fin = open(os.path.join(DATA_PATH,"text.test.txt"), "r")
for line in fin:
    fout.write(" ".join(sp.EncodeAsPieces(line)) + "\n")
fin.close()
fout.close()
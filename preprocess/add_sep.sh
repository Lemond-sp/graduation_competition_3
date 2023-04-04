for i in "train" "test" "dev"
do
sed -i s/"\s"/[SEP]/ "../../prepro/pseudo/text.${i}.txt"
done
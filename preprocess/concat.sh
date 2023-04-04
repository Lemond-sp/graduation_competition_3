for i in "train" "test" "dev"
do
paste "../../data/text.${i}.txt" "../../prepro/spm/en/${i}.en" > "../../prepro/spm/text.${i}.txt"
done
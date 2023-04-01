import argparse

def main(args):
    score = []
    # round
    with open(args.file_dir01) as f1,open(args.file_dir02) as f2:
        for s1,s2 in zip(f1,f2):
            avg = round((int(s1) + int(s2)) / 2 )
            score.append(avg)

    # output
    with open(args.res_dir,'w') as fw:
      for s in score:
        s = int(s)
        fw.write(str(s) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # train parameter
    parser.add_argument("--file_dir01", type=str, default="/home/kajikawa_r/competition/gradcomp/ch03/submission/l-lite/eval_test03.txt")
    parser.add_argument("--file_dir02", type=str, default="/home/kajikawa_r/competition/gradcomp/ch03/submission/l-lite/eval_test02.txt")
    parser.add_argument("--res_dir", type=str, default="/home/kajikawa_r/competition/gradcomp/ch03/submission/ensemble/eval_ensem.txt")
    args = parser.parse_args()
    main(args)
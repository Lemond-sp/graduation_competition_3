import argparse

def main(args):
    score = []
    # round
    with open(args.file_dir01) as f1,open(args.file_dir02) as f2,open(args.file_dir03) as f3,open(args.file_dir04) as f4:
        for s1,s2,s3,s4 in zip(f1,f2,f3,f4):
            avg = round((int(s1) + int(s2)+int(s3)+int(s4)) / 4 )
            score.append(avg)

    # output
    with open(args.res_dir,'w') as fw:
      for s in score:
        s = int(s)
        fw.write(str(s) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # train parameter
    parser.add_argument("--file_dir01", type=str, default="/home/kajikawa_r/competition/gradcomp/ch03/submission/large/large-exp01.txt")
    parser.add_argument("--file_dir02", type=str, default="/home/kajikawa_r/competition/gradcomp/ch03/submission/l-lite/large.txt")
    parser.add_argument("--file_dir03", type=str, default="/home/kajikawa_r/competition/gradcomp/ch03/submission/large-deberta/exp01.txt")
    parser.add_argument("--file_dir04", type=str, default="/home/kajikawa_r/competition/gradcomp/ch03/submission/large-waseda/exp01.txt")
    parser.add_argument("--res_dir", type=str, default="/home/kajikawa_r/competition/gradcomp/ch03/submission/ensemble/all_ensem.txt")
    args = parser.parse_args()
    main(args)
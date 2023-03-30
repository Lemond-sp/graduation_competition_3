import numpy as np
import os

# def main():
#     for i in ["label.dev.txt","label.train.txt","text.dev.txt","text.test.txt","text.train.txt"]:
#         load_data(i)

def load_data(path: str,data: str):
    with open(os.path.join(path,data)) as f:
        list_data = f.read().split('\n')
    del list_data[len(list_data) - 1]
    return list_data
# if __name__ == "__main__":
#     main()
import os
import math
import random


def read_txt(file):
    data = []
    for line in open(file, "r"):
        data.append(line)
    return data


def shuffle(items, size=0.01):
    tail_index = math.floor(len(items) * size)
    shuffled = sorted(items, key=lambda k: random.random())[:tail_index]
    return shuffled


def generate_test_data(file_cn, file_en):
    path_cn = os.path.join("data", file_cn)
    path_en = os.path.join("data", file_en)
    data_cn = read_txt(path_cn)
    data_en = read_txt(path_en)
    assert len(data_cn) == len(data_en)

    index = shuffle(list(range(len(data_cn))), 0.01)

    cn_test = [data_cn[idx] for idx in index]
    en_test = [data_en[idx] for idx in index]

    file_name_cn = file_cn.split(".")[0]
    file_name_en = file_en.split(".")[0]

    # cn
    with open("data/" + file_name_cn + ".test.txt", "w") as f:
        for line in cn_test:
            f.write(line)

    # en
    with open("data/" + file_name_en + ".test.txt", "w") as f:
        for line in en_test:
            f.write(line)


if __name__ == "__main__":
    generate_test_data("cn.txt", "en.txt")

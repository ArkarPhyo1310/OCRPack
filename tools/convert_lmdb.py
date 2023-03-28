import lmdb
import numpy as np


def split_from_file(gt_file: str, test_ratio: float = 0.2, val_ratio: float = 0.1):
    with open(gt_file, "r", encoding="utf-8") as f:
        data = f.read()

    word_list = data.split("\n")
    data_np = np.asarray(word_list)
    np.random.shuffle(data_np)

    train_index = int(len(data) * (1 - test_ratio))

    train_set = data_np[:train_index]
    test_set = data_np[train_index:]

    if val_ratio:
        train_index = int(len(train_set) * (1 - val_ratio))
        train_set = train_set[:train_index]
        val_set = train_set[train_index:]

        with open("./data/val.txt", "a", encoding="utf-8") as f:
            for content in val_set:
                f.write(content + "\n")

    with open("./data/train.txt", "a", encoding="utf-8") as f:
        for content in train_set:
            f.write(content + "\n")

    with open("./data/test.txt", "a", encoding="utf-8") as f:
        for content in test_set:
            f.write(content + "\n")


def convert_lmdb():
    pass

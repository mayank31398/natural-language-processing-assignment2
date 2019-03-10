import os

import numpy as np
import torch as th

from MyNet import ModelSaver, Evaluate
from test_utils import GetTestPoint, ReadData, ReadDictionary

TRAIN_PATH = os.path.join("Data", "dataset")
# CWD = os.getcwd()
CWD = os.path.join("Models", "Pad")


def Rank(x):
    x = 1 - x
    x = x.argsort()

    ranks = np.empty_like(x)
    ranks[x] = np.arange(x.shape[0])
    ranks = ranks + 1

    return ranks.tolist()


def List2String(x):
    for i in range(len(x)):
        x[i] = str(x[i])
    return " ".join(x)

def main():
    saver = ModelSaver(CWD)
    model, word_index, _, _ = saver.LoadCheckpoint("7.pt")

    sentences = ReadData(os.path.join(
        "Data", "eval_data.txt"), word_index, to_lower=True)
    dictionary = ReadDictionary(os.path.join(
        "Data", "eval_data.txt.td"), word_index)

    with open("output.txt", "w") as file:
        for i in range(len(sentences)):
            sentence = sentences[i]
            vocabulary = dictionary[i]

            dataloader = GetTestPoint(sentence, vocabulary)
            results = Evaluate(model, dataloader)

            ranks = Rank(results)
            ranks = List2String(ranks)

            file.write(ranks + "\n")


if(__name__ == "__main__"):
    main()

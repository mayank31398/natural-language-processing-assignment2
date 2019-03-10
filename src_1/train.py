import os
import pickle

import gensim
import numpy as np
import torch as th
import torch.functional as F
import torch.nn as nn
from gensim.models import KeyedVectors

from MyNet import ModelSaver, MyNet, test, train
from params import EMBEDDING_DIM, EPOCHS, LEARNING_RATE, SAVE_EVERY, TEST_EVERY
from utils import (BuildWord2Index, BuildWord2Vec, GetData, GetDataLoader,
                   LoadWord2Vec, SaveWord2Vec)

TRAIN_PATH = os.path.join("Data", "dataset")
CWD = os.getcwd()


def main():
    # os.makedirs(os.path.join("Models", "Word2Vec"), exist_ok=True)
    data, word_frequency = GetData(
        TRAIN_PATH, to_lower=True, remove_proper_nouns=True)
    # Not using unk embeddings yet
    # I am doing lowercase. Think about it!

    # Load Google's embeddings
    word2vec = KeyedVectors.load_word2vec_format(os.path.join(
        "Data", "GoogleNews-vectors-negative300.bin"), binary=True)

    # # Train word2vec
    # word2vec = BuildWord2Vec(os.path.join(
    #     "Data", "GoogleNews-vectors-negative300.bin"), data)

    # Get word_index dictionary
    word_index = BuildWord2Index(word_frequency.keys())

    train_loader, test_loader = GetDataLoader(data, word_index)

    # Set up the model
    model = MyNet(len(word_index), EMBEDDING_DIM, padding_idx=1)

    optimizer = th.optim.Adamax(
        model.GetTrainableParameters(), lr=LEARNING_RATE)
    scheduler = th.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.9)
    saver = ModelSaver(CWD)

    model.InitializeEmbeddings(word2vec, word_index, EMBEDDING_DIM)

    print("Train batches =", len(train_loader))
    print("Test batches =", len(test_loader))
    print("####################$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$####################\n")

    with open("out.txt", "w") as file:
        for epoch in range(EPOCHS):
            print("Epoch", epoch)

            loss, real_prob, fake_prob = train(model, train_loader, optimizer)

            print("Train loss =", loss)
            print("Train real prob =", real_prob)
            print("Train fake prob =", fake_prob)
            file.write("Train loss = " + str(loss) + "\n")
            file.write("Train real prob = " + str(real_prob) + "\n")
            file.write("Train fake prob = " + str(fake_prob) + "\n")

            scheduler.step()

            if(epoch % TEST_EVERY == 0):
                loss, real_prob, fake_prob = test(model, test_loader)

                print("Test loss =", loss)
                print("Test real prob =", real_prob)
                print("Test fake prob =", fake_prob)
                file.write("Test loss = " + str(loss) + "\n")
                file.write("Test real prob = " + str(real_prob) + "\n")
                file.write("Test fake prob = " + str(fake_prob) + "\n")

                print(
                    "####################$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$####################\n")

            if(epoch % 1 == 0):
                saver.SaveCheckpoint(model, word_index, optimizer,
                                     scheduler, str(epoch) + ".pt")


if(__name__ == "__main__"):
    main()

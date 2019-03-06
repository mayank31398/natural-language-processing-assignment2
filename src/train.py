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


def main():
    # os.makedirs(os.path.join("Models", "Word2Vec"), exist_ok=True)
    data, word_frequency = GetData(os.path.join(
        "Data", "dataset"), to_lower=True, remove_proper_nouns=True)
    # # Not using unk embeddings yet
    # # I am doing lowercase. Think about it!

    # # Train word2vec
    # word2vec = BuildWord2Vec(os.path.join(
    #     "Data", "GoogleNews-vectors-negative300.bin"), data)

    # # Save word2vec
    # SaveWord2Vec(os.path.join("Models", "Word2Vec", "word2vec.bin"), word2vec)

    # # Load word2vec
    # word2vec = LoadWord2Vec(os.path.join("Models", "Word2Vec", "word2vec.bin"))

    # # Get word_index dictionary
    word_index = BuildWord2Index(word_frequency.keys())
    # word_index = BuildWord2Index(word2vec.wv.vocab.keys())

    train_loader, test_loader = GetDataLoader(data, word_index)

    # # Set up the model
    model = MyNet(len(word_index), EMBEDDING_DIM)
    optimizer = th.optim.Adamax(model.parameters(), lr=LEARNING_RATE)
    scheduler = th.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.9)
    saver = ModelSaver(os.getcwd())

    print("Train batches =", len(train_loader))
    print("Test batches =", len(test_loader))
    print("####################$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$####################\n")

    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer)
        print("Train loss =", train_loss, "@ epoch", epoch)

        scheduler.step()

        if(epoch % TEST_EVERY == 0):
            test_loss = test(model, test_loader)
            print("Test loss =", test_loss, "@ epoch", epoch)
            print("####################$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$####################\n")

        if(epoch % 1 == 0):
            saver.SaveCheckpoint(model, word_index, optimizer,
                                 scheduler, str(epoch) + ".pt")
            model, word_index, _, _ = saver.LoadCheckpoint(
                model, str(epoch) + ".pt")


if(__name__ == "__main__"):
    main()

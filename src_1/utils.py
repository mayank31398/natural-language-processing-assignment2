import os
import pickle
import re

import gensim
import nltk
import numpy as np
import torch as th
from gensim.models import KeyedVectors, Word2Vec
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from params import BATCH_SIZE, CONTEXT, RANDOM_SEED, TEST_SIZE


def BuildWord2Vec(path, corpus):
    word2vec = Word2Vec(size=300, min_count=1)
    word2vec.build_vocab(corpus)

    pre_trained = KeyedVectors.load_word2vec_format(path, binary=True)
    num_examples = word2vec.corpus_count
    word2vec.build_vocab([list(pre_trained.vocab.keys())], update=True)
    word2vec.intersect_word2vec_format(path, binary=True, lockf=1)

    word2vec.train(corpus, total_examples=num_examples, epochs=word2vec.iter)

    return word2vec


def SaveWord2Vec(path, word2vec):
    with open(path, "wb") as file:
        pickle.dump(word2vec.wv, file)


def LoadWord2Vec(path):
    with open(path, "rb") as file:
        word2vec = pickle.load(file)
    return word2vec


def GetData(path, to_lower=False, remove_proper_nouns=False):
    files = os.listdir(path)
    corpus = ""
    for file in files:
        with open(os.path.join(path, file), "r") as file:
            file = file.readlines()
            for line in file:
                corpus += line

    corpus = re.sub("\n\n+", ". ", corpus)
    corpus = re.sub("\n", " ", corpus)
    tokenizer = nltk.PunktSentenceTokenizer()
    corpus = tokenizer.tokenize(corpus)

    data = []
    count_dict = {}
    for sentence in corpus:
        sentence = nltk.word_tokenize(sentence)
        if(remove_proper_nouns):
            sentence = RemoveProperNouns(sentence)
        sentence = RemovePunctuations(sentence, to_lower=to_lower)
        if(len(sentence) > 0):
            for word in sentence:
                if(word in count_dict):
                    count_dict[word] += 1
                else:
                    count_dict[word] = 1

            data.append(sentence)

    return data, count_dict


def RemoveProperNouns(sentence):
    pos = nltk.pos_tag(sentence)
    for i in range(len(pos)):
        if(pos[i][1] == "NNP" or pos[i][1] == "NNPS"):
            sentence[i] = "-pro-"
    return sentence


def RemovePunctuations(line, to_lower=False):
    filtered_line = []
    for word in line:
        if(len(word) <= 2 and not word.isalnum()):
            continue
        else:
            if(to_lower):
                word = word.lower()
                # to lowercase the word
            filtered_line.append(word)
    return filtered_line


def ReplaceWithUnk(data, count_threshold=None):
    if(count_threshold != None):
        count_dict = {}
        for word in data:
            if(word in count_dict):
                count_dict[word] += 1
            else:
                count_dict[word] = 1

        for i in range(len(data)):
            word = data[i]
            if(count_dict[word] < count_threshold):
                data[i] = "unk"

    return data


def PrintWord2Vec_dict(word2vec):
    for word in word2vec.wv.vocab.keys():
        print("{:.6f}".format(np.linalg.norm(
            word2vec[word])), str(word.encode("UTF-8")))


def BuildWord2Index(words):
    word_index = {}

    word_index["<<unk>>"] = 0
    word_index["<<pad>>"] = 1

    index = 2
    for word in words:
        word_index[word] = index
        index += 1

    return word_index


def Word2Index(data, word_index):
    for i in range(len(data)):
        data[i] = word_index[data[i]]
    return data


# This dataloader does padding
def GetDataLoader(data, word_index):
    def Pad(sentence):
        sentence = ["<<pad>>"] * CONTEXT + sentence + ["<<pad>>"] * CONTEXT
        return sentence

    def Get_XY_pairs(data, word_index):
        dataset_x = []
        dataset_y = []

        for sentence in data:
            # Pad each sentence
            sentence = Pad(sentence)

            # Convert to indices
            sentence = Word2Index(sentence, word_index)

            for i in range(len(sentence) - 2 * CONTEXT - 1):
                data_ = []
                data_ += sentence[i: i + CONTEXT]
                data_ += sentence[i + CONTEXT + 1: i + 2 * CONTEXT + 1]

                dataset_x.append(data_)
                dataset_y.append(sentence[i + CONTEXT])

        return dataset_x, dataset_y

    def MakeDataLoader(dataset_x, dataset_y):
        dataset_x = np.array(dataset_x).astype(np.long)
        dataset_y = np.array(dataset_y).astype(np.long)
        dataset_y = np.expand_dims(dataset_y, axis=1)

        dataset = TensorDataset(
            th.tensor(dataset_x).type(th.LongTensor),
            th.tensor(dataset_y).type(th.LongTensor)
        )

        data_loader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=BATCH_SIZE,
            drop_last=False
        )

        return data_loader

    dataset_x, dataset_y = Get_XY_pairs(data, word_index)

    dataset_x_train, dataset_x_test, dataset_y_train, dataset_y_test = train_test_split(
        dataset_x, dataset_y, random_state=RANDOM_SEED, test_size=TEST_SIZE)

    train_loader = MakeDataLoader(dataset_x_train, dataset_y_train)
    test_loader = MakeDataLoader(dataset_x_test, dataset_y_test)

    return train_loader, test_loader

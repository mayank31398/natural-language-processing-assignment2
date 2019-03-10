import nltk
import numpy as np
import torch as th
from torch.utils.data import DataLoader, TensorDataset

from params import BATCH_SIZE, CONTEXT


def UpdateWord2Index(word_index, word2vec):
    index = len(word_index)
    for word in word2vec.wv.vocab.keys():
        if(word not in word_index):
            word_index[word] = index
            index += 1
    return word_index


def ReadData(path, word_index, to_lower=False):
    def Pad(sentence):
        sentence = ["<<pad>>"] * CONTEXT + sentence + ["<<pad>>"] * CONTEXT
        return sentence

    def Sentence2Index(sentence, word_index):
        x = []
        for i in range(len(sentence)):
            word = sentence[i]
            if(word in word_index):
                index = word_index[word]
                x.append(index)
            else:
                x.append(0)
        return x

    def RemoveProperNouns(sentence):
        pos = nltk.pos_tag(sentence)
        for i in range(len(pos)):
            if((pos[i][1] == "NNP" or pos[i][1] == "NNPS") and pos[i][0] != "<<target>>"):
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

    data = []
    with open(path, "r") as file:
        for line in file:
            sentence = line.split("::::")[0]
            sentence = sentence.split()
            # sentence = nltk.word_tokenize(sentence)

            sentence = RemoveProperNouns(sentence)
            sentence = RemovePunctuations(sentence)
            # sentence = RemovePunctuations(sentence, to_lower=to_lower)

            # Pad sentences
            sentence = Pad(sentence)

            index = sentence.index("<<target>>")
            sentence = sentence[index - CONTEXT: index] + \
                sentence[index + 1: index + CONTEXT + 1]

            sentence = Sentence2Index(sentence, word_index)
            data.append(sentence)

    return data


def ReadDictionary(path, word_index):
    def Vocabulary2Index(vocabulary, word_index):
        y = []
        for i in range(len(vocabulary)):
            word = vocabulary[i]
            if(word in word_index):
                index = word_index[word]
                y.append(index)
            else:
                y.append(0)
        return y

    dictionary = []
    with open(path, "r") as file:
        for line in file:
            line = line.split()
            line = Vocabulary2Index(line, word_index)
            dictionary.append(line)
    return dictionary


def GetTestPoint(sentence, vocabulary):
    sentence = [sentence] * len(vocabulary)

    x = np.array(sentence)
    y = np.array(vocabulary).reshape(x.shape[0], 1)

    dataset = TensorDataset(
        th.tensor(x).type(th.LongTensor),
        th.tensor(y).type(th.LongTensor)
    )

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        drop_last=False,
        batch_size=BATCH_SIZE
    )

    return dataloader

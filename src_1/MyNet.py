import os

import numpy as np
import torch as th
import torch.nn as nn
from tqdm import tqdm

import params


class MyNet(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=1):
        super().__init__()

        self.embeddings = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx)  # 8, 300

        # Embeddings = ET + delta
        self.T = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.delta = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx)  # 8, 300

        # self.T = nn.Sequential(
        #     nn.Linear(embedding_dim, embedding_dim), nn.SELU())

        # 1, 2400
        self.c = nn.Sequential(
            nn.Conv1d(1, 2, 7), nn.MaxPool1d(2), nn.SELU(),  # 2, 1197
            nn.Conv1d(2, 4, 7, stride=2), nn.SELU(),  # 4, 596
            nn.Conv1d(4, 8, 5), nn.MaxPool1d(2), nn.SELU(),  # 8, 296
            nn.Conv1d(8, 16, 5), nn.MaxPool1d(2), nn.SELU(),  # 16, 146
            nn.Conv1d(16, 32, 5), nn.MaxPool1d(2), nn.SELU(),  # 32, 71
            nn.Conv1d(32, 64, 5, stride=2), nn.SELU(),  # 64, 34
            nn.Conv1d(64, 128, 3), nn.MaxPool1d(2), nn.SELU()  # 128, 16
        )

        # 1, 300
        self.w = nn.Sequential(
            nn.Conv1d(1, 8, 5), nn.MaxPool1d(2), nn.SELU(),  # 8, 146
            nn.Conv1d(8, 32, 5), nn.MaxPool1d(2), nn.SELU(),  # 32, 71
            nn.Conv1d(32, 64, 5, stride=2), nn.SELU(),  # 64, 34
            nn.Conv1d(64, 128, 3), nn.MaxPool1d(2), nn.SELU()  # 128, 16
        )

        # 4096
        self.l = nn.Sequential(
            nn.Linear(4096, 1024), nn.SELU(),  # 1024
            nn.Linear(1024, 256), nn.SELU(),  # 256
            nn.Linear(256, 64), nn.SELU(),  # 64
            nn.Linear(64, 16), nn.SELU(),  # 16
            nn.Linear(16, 4), nn.SELU(),  # 4
            nn.Linear(4, 1), nn.Sigmoid()  # 1
        )

    # def GetEmbeddings(self, x):
    #     return self.T(self.embeddings(x))

    def GetEmbeddings(self, x):
        E = self.T(self.embeddings(x))
        E = E + self.delta(x)

        return E

    def forward(self, x, y):
        c = self.GetEmbeddings(x)
        c = c.view(x.shape[0], 1, -1)

        w = self.GetEmbeddings(y)
        w = w.view(x.shape[0], 1, -1)

        c = self.c(c)
        w = self.w(w)

        cw = th.cat([c, w], dim=1)
        cw = cw.view(x.shape[0], -1)

        out = self.l(cw)

        return out

    def Evaluate(self, x, y):
        c = x.view(x.shape[0], 1, -1)
        w = y.view(x.shape[0], 1, -1)

        c = self.c(c)
        w = self.w(w)

        cw = th.cat([c, w], dim=1)
        cw = cw.view(x.shape[0], -1)

        out = self.l(cw)

        return out

    def GetTrainableParameters(self):
        trainable_parameters = []
        for i in self.parameters():
            trainable_parameters.append(i)
        # Remove embedding layer
        trainable_parameters = trainable_parameters[1:]

        return trainable_parameters

    def InitializeEmbeddings(self, word2vec, word_index, embedding_dim):
        num_embeddings = len(word_index)

        xavier = np.sqrt(6 / embedding_dim)
        weights = np.random.uniform(-xavier, xavier,
                                    size=(num_embeddings, embedding_dim))
        weights[self.embeddings.padding_idx, :] = 0

        for word in word_index:
            if(word in word2vec.wv.vocab.keys()):
                index = word_index[word]
                weights[index, :] = word2vec[word]

        self.embeddings.weight = nn.Parameter(th.Tensor(weights))
        self.embeddings.weight.requires_grad = False


def train(model: MyNet, data_loader, optimizer):
    model.cuda()
    model.train()

    total_loss = 0
    total_fake_prob = 0
    total_real_prob = 0
    batches = len(data_loader)

    for data_x, data_y in tqdm(data_loader):
        batch_size = data_x.shape[0]

        data_x = data_x.cuda()
        data_y = data_y.cuda()

        # Drop some words
        probabilities = th.ones(data_x.shape) * 0.9
        probabilities = th.bernoulli(probabilities).cuda().type(th.long)
        data_x = data_x * probabilities

        out_true = model(data_x, data_y)

        fake_y = th.randperm(data_x.shape[0])
        fake_y = data_y[fake_y]
        out_fake = model(data_x, fake_y)

        loss = -th.log(out_true) - th.log(1 - out_fake)
        loss = loss.sum() / batch_size

        out_fake = out_fake.sum() / batch_size
        out_true = out_true.sum() / batch_size

        # Do backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_real_prob += out_true.item()
        total_fake_prob += out_fake.item()

    avg_loss = total_loss / batches
    avg_real_prob = total_real_prob / batches
    avg_fake_prob = total_fake_prob / batches

    return avg_loss, avg_real_prob, avg_fake_prob


def test(model: MyNet, data_loader):
    model.cuda()
    model.eval()

    total_loss = 0
    total_fake_prob = 0
    total_real_prob = 0
    batches = len(data_loader)

    for data_x, data_y in tqdm(data_loader):
        batch_size = data_x.shape[0]

        data_x = data_x.cuda()
        data_y = data_y.cuda()

        out_true = model(data_x, data_y)

        fake_y = th.randperm(data_x.shape[0])
        fake_y = data_y[fake_y]
        out_fake = model(data_x, fake_y)

        loss = -th.log(out_true) - th.log(1 - out_fake)
        loss = loss.sum() / batch_size

        out_fake = out_fake.sum() / batch_size
        out_true = out_true.sum() / batch_size

        total_loss += loss.item()
        total_real_prob += out_true.item()
        total_fake_prob += out_fake.item()

    avg_loss = total_loss / batches
    avg_real_prob = total_real_prob / batches
    avg_fake_prob = total_fake_prob / batches

    return avg_loss, avg_real_prob, avg_fake_prob


def Evaluate(model: MyNet, data_loader):
    model.cuda()
    model.eval()

    results = []
    for data_x, data_y in data_loader:
        data_x = data_x.cuda()
        data_y = data_y.cuda()

        out = model.Evaluate(data_x, data_y)
        out = out.detach().cpu().numpy()
        results.append(out)

    results = np.concatenate(results, axis=0)
    results = np.squeeze(results)

    return results


class ModelSaver:
    def __init__(self, path):
        self.path = path

    def SaveCheckpoint(
        self,
        model,
        word_index,
        optimizer,
        scheduler,
        filename
    ):
        model_dict = model.state_dict()
        optimizer_dict = optimizer.state_dict()
        scheduler_dict = scheduler.state_dict()

        param_dict = {
            "BATCH_SIZE": params.BATCH_SIZE,
            "CONTEXT": params.CONTEXT,
            "EMBEDDING_DIM": params.EMBEDDING_DIM,
            "EPOCHS": params.EPOCHS,
            "LEARNING_RATE": params.LEARNING_RATE,
            "RANDOM_SEED": params.RANDOM_SEED,
            "TEST_EVERY": params.TEST_EVERY,
            "TEST_SIZE": params.TEST_SIZE,
            "SAVE_EVERY": params.SAVE_EVERY
        }

        checkpoint = {
            "model_dict": model_dict,
            "optimizer_dict": optimizer_dict,
            "scheduler_dict": scheduler_dict,
            "param_dict": param_dict,
            "word_index": word_index
        }

        th.save(checkpoint, os.path.join(self.path, filename))

    def LoadCheckpoint(self, filename, optimizer=None, scheduler=None):
        checkpoint = th.load(os.path.join(self.path, filename))

        model = MyNet(len(checkpoint["word_index"]),
                      checkpoint["param_dict"]["EMBEDDING_DIM"])

        model.load_state_dict(checkpoint["model_dict"])
        word_index = checkpoint["word_index"]
        if(optimizer != None):
            optimizer.load_state_dict(checkpoint["optimizer_dict"])
        if(scheduler != None):
            scheduler.load_state_dict(checkpoint["scheduler_dict"])

        return model, word_index, optimizer, scheduler

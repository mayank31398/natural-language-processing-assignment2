import os

import torch as th
import torch.nn as nn
from tqdm import tqdm

import params


class MyNet(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)  # 8, 300
        # self.embeddings.weight = nn.Parameter(trained_embeddings)

        # 1, 2400
        self.c = nn.Sequential(
            nn.Conv1d(1, 2, 7), nn.MaxPool1d(2), nn.SELU(),  # 2, 1197
            nn.Conv1d(2, 2, 7, stride=2), nn.SELU(),  # 2, 596
            nn.Conv1d(2, 4, 5), nn.MaxPool1d(2), nn.SELU(),  # 4, 296
            nn.Conv1d(4, 4, 5), nn.MaxPool1d(2), nn.SELU(),  # 4, 146
            nn.Conv1d(4, 8, 5), nn.MaxPool1d(2), nn.SELU(),  # 8, 71
            nn.Conv1d(8, 8, 5, stride=2), nn.SELU(),  # 8, 34
            nn.Conv1d(8, 16, 3), nn.MaxPool1d(2), nn.SELU()  # 16, 16
        )

        # 1, 300
        self.w = nn.Sequential(
            nn.Conv1d(1, 2, 5), nn.MaxPool1d(2), nn.SELU(),  # 2, 146
            nn.Conv1d(2, 4, 5), nn.MaxPool1d(2), nn.SELU(),  # 4, 71
            nn.Conv1d(4, 8, 5, stride=2), nn.SELU(),  # 8, 34
            nn.Conv1d(8, 16, 3), nn.MaxPool1d(2), nn.SELU()  # 16, 16
        )

        # 512
        self.l = nn.Sequential(
            nn.Linear(512, 128), nn.SELU(),  # 128
            nn.Linear(128, 16), nn.SELU(),  # 16
            nn.Linear(16, 4), nn.SELU(),  # 4
            nn.Linear(4, 1), nn.Sigmoid()  # 1
        )

    def forward(self, x, y):
        c = self.embeddings(x)
        c = c.view(x.shape[0], 1, -1)

        w = self.embeddings(y)
        w = w.view(x.shape[0], 1, -1)
        
        c = self.c(c)
        w = self.w(w)

        cw = th.cat([c, w], dim=1)
        cw = cw.view(x.shape[0], -1)

        out = self.l(cw)

        return out


def train(model: MyNet, data_loader, optimizer):
    model.cuda()
    model.train()

    total_loss = 0
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

        # Do backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / batches

    return avg_loss


def test(model: MyNet, data_loader):
    model.cuda()
    model.eval()

    total_loss = 0
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

        total_loss += loss.item()

    avg_loss = total_loss / batches

    return avg_loss


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

    def LoadCheckpoint(self, model, filename, optimizer=None, scheduler=None):
        checkpoint = th.load(os.path.join(self.path, filename))

        model.load_state_dict(checkpoint["model_dict"])
        word_index = checkpoint["word_index"]
        if(optimizer != None):
            optimizer.load_state_dict(checkpoint["optimizer_dict"])
        if(scheduler != None):
            scheduler.load_state_dict(checkpoint["scheduler_dict"])

        return model, word_index, optimizer, scheduler

import json
import os
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


def readData(filepath, num=None):
    all_data = pd.read_csv(filepath)
    all_data["entireSentence"] = all_data["currentSentence"] + all_data["nextSentence"]
    if num == None:
        return all_data["currentSentence"].tolist(), all_data["nextSentence"].tolist(), \
               all_data["sentenceLabels"].tolist()
    else:
        return all_data["currentSentence"].tolist()[:num], all_data["nextSentence"].tolist()[:num], \
               all_data["sentenceLabels"].tolist()[:num]


def buildWord2Index(texts):
    if os.path.exists(os.path.join("data", "lzjTrain", "word2index.json")):
        with open(os.path.join("data", "lzjTrain", "word2index.json"), "r", encoding="utf-8") as f:
            word2index = json.load(f)
    else:
        word2index = {"UNK": 0, "CLS": 1, "SEP": 2, "PAD": 3, "MASK": 4, "UNSED": 1}
        for text in texts:
            for word in text:
                word2index[word] = word2index.get(word, len(word2index))
        with open(os.path.join("data", "lzjTrain", "word2index.json"), "w", encoding="utf-8") as f:
            json.dump(word2index, f)
    return word2index


def loadConfig(texts):
    if os.path.exists(os.path.join("data", "lzjTrain", "config.json")):
        with open(os.path.join("data", "lzjTrain", "config.json"), "r") as f:
            config = json.load(f)
    else:
        word2index = buildWord2Index(texts)
        config = {
            "epoch": 10,
            "lr": 1e-4,
            "train_batch": 10,
            "val_batch": 5,
            "maxLen": 512,
            "headNum": 8,
            "embeddingNum": 768,
            "word2index": word2index,
            "feedForwardDim": 128,
            "layerNum": 5,
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        }
        with open(os.path.join("data", "lzjTrain", "config.json"), "w") as f:
            json.dump(config, f)
    return config


class bertDataset(Dataset):
    def __init__(self, currentSentences, nextSentences, labels, config):
        super(bertDataset, self).__init__()
        self.currentSentences = currentSentences
        self.nextSentences = nextSentences
        self.sentenceLabels = labels
        self.config = config

    def __getitem__(self, item):
        CurrentSentenceIndex = [self.config["word2index"].get(current, 0) for current in
                                self.currentSentences[item][:250]]
        NextSentenceIndex = [self.config["word2index"].get(next, 0) for next in self.nextSentences[item][:250]]
        TextIndex = CurrentSentenceIndex + [self.config["word2index"]["SEP"]] + NextSentenceIndex
        SegmentIndex = [0] * (len(CurrentSentenceIndex) + 1) + [1] * (len(NextSentenceIndex))
        label = self.sentenceLabels[item]
        TextLen = len(TextIndex)
        mask2index = [0] * len(TextIndex)
        for i in range(TextLen):
            if i == len(CurrentSentenceIndex) + 1:
                continue
            if random.random() < 0.15:
                if random.random() < 0.8:
                    mask2index[i] = self.config["word2index"]["MASK"]
                elif random.random() < 0.9:
                    mask2index[i] = random.choice(range(5, len(self.config["word2index"])))
                else:
                    mask2index[i] = TextIndex[i]
        return TextIndex, SegmentIndex, mask2index, label, TextLen

    def dataProcess(self, datas):
        batchIndexs, batchSegments, batchMasks, batchLabels, batchLens = zip(*datas)
        bMaxLen = max(batchLens)
        PadBatchIndexs = np.array([[self.config["word2index"]["CLS"]] + idx + [self.config["word2index"]["SEP"]] +
                                   [self.config["word2index"]["PAD"]] * (bMaxLen - len(idx)) for idx in batchIndexs])
        PadBatchSegments = np.array([[0] + seg + [1] * (bMaxLen - len(seg) + 1) for seg in batchSegments])
        PadBatchMasks = np.array([[0] + idx + [0] + [0] * (bMaxLen - len(idx)) for idx in batchMasks])

        return torch.from_numpy(PadBatchIndexs), torch.from_numpy(PadBatchSegments), torch.from_numpy(
            PadBatchMasks), torch.tensor(batchLabels), bMaxLen + 2

    def __len__(self):
        return len(self.sentenceLabels)


class bertEmbedding(nn.Module):
    def __init__(self, config):
        super(bertEmbedding, self).__init__()
        self.tokenEmbedding = nn.Embedding(len(config["word2index"]), 768)
        self.segmentEmbedding = nn.Embedding(2, 768)
        self.positionEmbedding = nn.Embedding(config["maxLen"], 768)

    def forward(self, datas, segments, bLen):
        tokenEmebedding = self.tokenEmbedding(datas)
        segmentEmbedding = self.segmentEmbedding(segments)
        positionEmbedding = self.positionEmbedding(torch.arange(0, bLen, device=config["device"]))
        embedding = tokenEmebedding + segmentEmbedding + positionEmbedding.repeat(segmentEmbedding.shape[0], 1, 1)
        return embedding


class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        self.Q = nn.Linear(config["embeddingNum"], int(config["embeddingNum"] / config["headNum"]))
        self.K = nn.Linear(config["embeddingNum"], int(config["embeddingNum"] / config["headNum"]))
        self.V = nn.Linear(config["embeddingNum"], int(config["embeddingNum"] / config["headNum"]))
        self.SoftMax = nn.Softmax(dim=-1)

    def forward(self, embedding):
        q = self.Q(embedding)
        k = self.K(embedding)
        v = self.V(embedding)
        qk_score = self.SoftMax(
            (q @ torch.permute(k, (0, 2, 1))) / (pow(config["embeddingNum"] / config["headNum"], 1 / 2)))
        z = qk_score @ v
        return z


class MutiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head = config["headNum"]
        self.QKV_score = nn.Sequential(*[SelfAttention(config) for i in range(config["headNum"])])

    def forward(self, datas):
        QKV = [self.QKV_score[h](datas) for h in range(self.head)]
        MHA_score = torch.concat(tuple(QKV), dim=-1)
        return MHA_score


class AddNorm(nn.Module):
    def __init__(self, config):
        super(AddNorm, self).__init__()
        self.layerNorm = nn.LayerNorm(config["embeddingNum"])
        self.dropout = nn.Dropout(0.1)

    def forward(self, history, current):
        add = history + current
        add_drop = self.dropout(add)
        layernorm = self.layerNorm(add_drop)
        return layernorm


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Linear(config["embeddingNum"], config["feedForwardDim"]),
            nn.GELU(),
            nn.Linear(config["feedForwardDim"], config["embeddingNum"]))

    def forward(self, datas):
        feed = self.sequence(datas)
        return feed


class BackBone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.MutiHead = MutiHeadAttention(config)
        self.AddNorm1 = AddNorm(config)
        self.FeedForward = FeedForward(config)
        self.AddNorm2 = AddNorm(config)

    def forward(self, embedding):
        attention = self.MutiHead(embedding)
        addNorm1 = self.AddNorm1(embedding, attention)
        feedForward = self.FeedForward(addNorm1)
        addNorm2 = self.AddNorm2(addNorm1, feedForward)
        return addNorm2


class Pooler(nn.Module):
    def __init(self):
        super(Pooler, self).__init()

    def forward(self, poolIn):
        poolerOut = poolIn[:, 0, :]
        return poolerOut


class bertModel(nn.Module):
    def __init__(self, config):
        super(bertModel, self).__init__()
        self.embedding = bertEmbedding(config)
        self.backbone = nn.Sequential(*[BackBone(config) for i in range(config["layerNum"])])
        self.pooler = Pooler()

    def forward(self, datas, segments, bLen):
        embedding = self.embedding(datas, segments, bLen)
        for i in range(config["layerNum"]):
            embedding = self.backbone(embedding)
        pooler = self.pooler(embedding)
        return embedding, pooler


class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.bert = bertModel(config)
        self.mlmTask = nn.Linear(config["embeddingNum"], len(config["word2index"]))
        self.nspTask = nn.Linear(config["embeddingNum"], 2)
        self.mlmLoss = nn.CrossEntropyLoss(ignore_index=0)
        self.nspLoss = nn.CrossEntropyLoss()

    def forward(self, datas, segments, bLen, masks=None, labels=None):
        seqOut, senOut = self.bert(datas, segments, bLen)
        mlmTask = self.mlmTask(seqOut)
        nspTask = self.nspTask(senOut)
        if masks is not None and labels is not None:
            mlmLoss = self.mlmLoss(mlmTask.reshape(-1, mlmTask.shape[-1]), masks.reshape(-1).long())
            nspLoss = self.nspLoss(nspTask, labels)
            loss = mlmLoss + nspLoss
            return loss
        else:
            mlmPre = torch.argmax(mlmTask, dim=-1)
            nspPre = torch.argmax(nspTask, dim=-1)
            return mlmPre, nspPre


if __name__ == "__main__":
    currentSentences, nextSentences, sentenceLabels = readData(os.path.join("data", "lzjTrain", "NspDatasets.csv"),
                                                               num=20000)
    assert len(currentSentences) == len(sentenceLabels) and len(nextSentences) == len(sentenceLabels), \
        "NspDataset数据集长度不一致！"
    config = loadConfig(currentSentences)

    trainDataset = bertDataset(currentSentences[:-2000], nextSentences[:-2000], sentenceLabels[:-2000], config)
    trainDataloader = DataLoader(trainDataset, batch_size=config["train_batch"], shuffle=True,
                                 collate_fn=trainDataset.dataProcess)

    valDataset = bertDataset(currentSentences[-2000:], nextSentences[-2000:], sentenceLabels[-2000:], config)
    valDataloader = DataLoader(valDataset, batch_size=config["val_batch"], shuffle=False,
                               collate_fn=valDataset.dataProcess)

    BTmodel = MyModel(config).to(config["device"])
    optimizer = torch.optim.Adam(BTmodel.parameters(), config["lr"])

    for e in range(config["epoch"]):
        BTmodel.train()
        for trainBatchIndexs, trainBatchSegments, trainBatchMasks, trainBatchLabels, trainBatchLen in tqdm(
                trainDataloader):
            trainBatchIndexs, trainBatchSegments, trainBatchMasks, trainBatchLabels = trainBatchIndexs.to(
                config["device"]), trainBatchSegments.to(config["device"]), trainBatchMasks.to(
                config["device"]), trainBatchLabels.to(config["device"])
            loss = BTmodel.forward(trainBatchIndexs, trainBatchSegments, trainBatchLen, trainBatchMasks,
                                   trainBatchLabels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            break

        BTmodel.eval()
        mlmNum, mlmRight, nspRight = 0, 0, 0
        for valBatchIndexs, valBatchSegments, valBatchMasks, valBatchLabels, valBatchLen in tqdm(valDataloader):
            valBatchIndexs, valBatchSegments, valBatchMasks, valBatchLabels = valBatchIndexs.to(
                config["device"]), valBatchSegments.to(config["device"]), valBatchMasks.to(
                config["device"]), valBatchLabels.to(config["device"])
            mlm, nsp = BTmodel.forward(valBatchIndexs, valBatchSegments, valBatchLen)
            mlmNum += len(mlm[valBatchMasks != 0])
            mlmRight += torch.sum(mlm[valBatchMasks != 0] == valBatchMasks[valBatchMasks != 0])
            nspRight += torch.sum(nsp == valBatchLabels)
            break

        print(f"mlmAcu:{mlmRight / mlmNum * 100:.4f}%\tnspAcu:{nspRight / len(valDataset) * 100:.4f}%")

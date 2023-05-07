import os
import re
import random
import pandas as pd


def splitDatas(data):
    text = data.replace("\xad\u3000\u3000", "")
    pattern = r"[，；。？！?!]"
    split_data = re.split(pattern, text)
    dataList = combineData(split_data)
    return dataList


def combineData(split_data):
    paragraph, sentence = [], ""
    for data in split_data:
        if len(data) < 5:
            continue
        elif len(data) < 25:
            sentence += data + "，"
            if len(sentence) < 50 or random.random() > 0.6:
                continue
            else:
                paragraph.append(sentence[:-1] + "。")
                sentence = ""
        else:
            paragraph.append(data + "！")
    return paragraph


def readDatas(filepath):
    datas = pd.read_csv(filepath, encoding="utf-8", index_col=False)["content"].tolist()
    paragraphList = []
    for data in datas:
        paragraphList.append(splitDatas(data))
    return paragraphList


# 构建 IsNext 和 NotNext
def buildNsp(paragraphs):
    currentSentence, nextSentence, sentenceLabels = [], [], []
    for paragraph in paragraphs:
        if len(paragraph) < 3:
            continue
        for textId, text in enumerate(paragraph[:-1]):
            currentSentence.extend([text] * 2)
            nextSentence.append(paragraph[textId + 1])
            while True:
                notNext = random.choice(paragraph)
                if notNext != text and notNext != paragraph[textId + 1]:
                    nextSentence.append(notNext)
                    break
            sentenceLabels.extend([1, 0])
    nspDataset = {
        "currentSentence": currentSentence,
        "nextSentence": nextSentence,
        "sentenceLabels": sentenceLabels
    }
    Nsp = pd.DataFrame(nspDataset)
    Nsp.to_csv(os.path.join("data", "lzjTrain", "NspDatasets.csv"))
    return Nsp


if __name__ == "__main__":
    # 每句话长度不超过512
    paragraphs = readDatas(os.path.join("data", "origin_data.csv"))
    NspDatas = buildNsp(paragraphs)

    print("")

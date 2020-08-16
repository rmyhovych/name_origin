import torch

import random

import matplotlib.pyplot as plt

import os
import re

import nn


def charOneHot(c, chars: list):
    oh = torch.zeros(size=(len(chars),))
    oh[chars.index(c)] = 1
    return oh


def wordToOneHot(word: str, chars: list):
    return [charOneHot(c, chars) for c in word]


def oneHotToWord(oneHot, chars: list):
    w = []
    for i in oneHot:
        m = torch.argmax(i)
        w.append(chars[int(m.item())])

    return "".join(w)


def getFilepaths(dirname):
    return [
        os.path.join(dirname, f)
        for f in os.listdir(dirname)
        if os.path.isfile(os.path.join(dirname, f))
    ]


def readNameData(dirname):
    files = getFilepaths(dirname)
    chars = set()

    rawNames = []
    for label, fname in enumerate(files):
        lines = []
        with open(fname, "r", encoding="utf8") as f:
            lines = f.readlines()

        namesInFile = []
        for nWord in lines:
            nWord = nWord.lower()
            nWord = re.sub(r"\W+", "", nWord)

            for c in nWord:
                chars.add(c)

            namesInFile.append(nWord)

        rawNames.append(namesInFile)

    chars = list(chars)

    data = []

    sizePerLang = min(len(l) for l in rawNames)
    for i, nameList in enumerate(rawNames):
        for w in nameList[:sizePerLang]:
            label = torch.zeros(size=(len(files),))
            label[i] = 1.0
            data.append((wordToOneHot(w, chars), label))

    return files, chars, data


def predictName(net, name):
    net.reset()
    y = None
    for charData in name:
        y = net(charData)

    return y


def validateTest(net, data):

    testLoss = 0.0
    testAccuracy = 0.0
    for nameData in data:
        name = nameData[0]
        label = nameData[1]

        y: torch.tensor = predictName(net, name)

        actualItem = torch.argmax(y).item()
        labelItem = torch.argmax(label).item()
        testAccuracy += 1 if actualItem == labelItem else 0

        loss = lossFunction(y, label)
        testLoss += loss.item()

    return testLoss / len(data), testAccuracy / len(data)


if __name__ == "__main__":

    files, chars, data = readNameData("names")

    net = nn.NetworkLSTM(len(chars), 2 * len(chars), len(files), torch.sigmoid)
    params = list(net.parameters())
    # print(params)

    optimizer = torch.optim.Adam(params, lr=0.001)
    lossFunction = torch.nn.MSELoss()

    random.shuffle(data)
    dataToUse = data[: int(1 * len(data))]

    trainingSet = dataToUse[: int(0.9 * len(dataToUse))]
    testSet = dataToUse[len(trainingSet) :]

    print(
        "dataToUse[{}] trainingSet[{}] testSet[{}]".format(
            len(dataToUse), len(trainingSet), len(testSet)
        )
    )

    nEpisodes = 200
    batchSize = 20

    losses = []
    testLosses = []
    testAccuracies = []
    for ep in range(nEpisodes):
        random.shuffle(trainingSet)

        if ep % 10 == 0:
            print(ep)
            testLoss, testAccuracy = validateTest(net, testSet)
            testLosses.append(testLoss)
            testAccuracies.append(testAccuracy)

        lossAverage = 0.0

        i = 0
        for i in range(0, len(trainingSet), batchSize):
            batch = trainingSet[i : i + batchSize]

            optimizer.zero_grad()

            for nameData in batch:
                name = nameData[0]
                label = nameData[1]

                y = predictName(net, name)

                loss = lossFunction(y, label)
                loss.backward(retain_graph=True)

                lossAverage += loss.item()

            optimizer.step()

        losses.append(lossAverage / (i * len(batch)))

    print("\n\n\tTRAINING DATA :")

    loss, accuracy = validateTest(net, trainingSet[:50])

    testingTraining = trainingSet[:30]
    for nameData in testingTraining:
        name = nameData[0]
        label = nameData[1]

        y = predictName(net, name)
        print(
            "{} : {}".format(
                oneHotToWord(name, chars), files[torch.argmax(label).item()]
            )
        )

    print("\n\n\tTEST DATA :")
    for nameData in testSet:
        name = nameData[0]
        label = nameData[1]

        y = predictName(net, name)
        print(
            "{} : {}".format(
                oneHotToWord(name, chars), files[torch.argmax(label).item()]
            )
        )

    plt.plot(losses)
    plt.show()

    plt.plot(testLosses)
    plt.show()

    plt.plot(testAccuracies)
    plt.show()

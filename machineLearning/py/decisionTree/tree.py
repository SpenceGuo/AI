import operator

from math import log
from typing import List


def createDataset():
    """
    构造数据集和类别标签
    :return: 返回数据集和类别标签
    """
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calShannonEnt(dataSet: List):
    """
    计算香农熵
    :param dataSet: 输入数据集
    :return: 返回香农熵
    """
    # 数据集中实例总数
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataset(dataSet: List, axis: int, value: int):
    """
    划分数据集
    :param dataSet:
    :param axis:
    :param value:
    :return:
    """
    retDataset = []
    for featVec in dataSet:
        # 将符合特征的数据抽取出来
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataset.append(reduceFeatVec)
    return retDataset


def chooseBestFeatureToSplit(dataSet: List):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataset = splitDataset(dataSet, i, value)
            prob = len(subDataset)/float(len(dataSet))
            newEntropy += prob * calShannonEnt(subDataset)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList: List):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    myDat, myLabels = createDataset()
    print(myDat)
    print(myLabels)
    print(calShannonEnt(myDat))
    print(chooseBestFeatureToSplit(myDat))

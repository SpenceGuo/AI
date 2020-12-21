import numpy as np
import operator


def createDateset():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


group, labels = createDateset()


def classify0(inX, dataSet, labels, k):
    """
    参数说明
    inX:用于分类的输入向量
    dataSet:输入的训练样本集
    labels:标签向量
    k:用于选择的最近邻居的数目

    return:输入向量inX的预测类别
    """
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, [dataSetSize, 1]) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    sortedDistIndicies = distance.argsort()

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


print(classify0([0, 0], group, labels, 3))

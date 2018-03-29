#!/home/louplus/env/bin python
# encoding: utf-8

from itertools import *
import numpy as np
import operator, math



def splitDataSet(dataset, axis, values):
    retDataSet = []
    if len(values) < 2:
        for featVec in dataset:
            if featVec[axis] == values[0]:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
    else:
        for featVec in dataset:
            for value in values:
                if featVec[axis] == value:
                    retDataSet.append(featVec)
    return retDataSet

def featuresplit(features):
    count = len(features)
    if count < 2:
        #print('check sample features, only one feature value')
        return ((features[0], ), )
    combinationsList = []
    resList = []
    for i in range(1, count):
        temp_combination = list(combinations(features, len(features[0:i])))
        combinationsList.extend(temp_combination)
    combiLen = len(combinationsList)
    return zip(combinationsList[0:combiLen//2],
            combinationsList[-1:combiLen//2-1:-1])

def chooseBestFeatrueToSplit(dataset):
    numFeatures = len(dataset[0][:-1])
    bestStDev = np.inf; bestFeature = -1; bestBinarySplit = ()
    for i in range(numFeatures):
        featList = [example[i] for example in dataset]
        uniqueVals = list(set(featList))
        for split in featuresplit(uniqueVals):
            stDev =0.0
            if len(split) == 1:
                #print('len(split) = 1, splib:\n', split)
                continue
            (left, right) = split
            left_subdataset = splitDataSet(dataset, i, left)
            #left_prob = len(left_subdataset) / len(dataset)
            S, u = calcstDev(left_subdataset)
            #stDev += left_prob * S
            stDev += S

            right_subDataSet = splitDataSet(dataset, i, right)
            #right_prob = len(right_subDataSet) / len(dataset)
            S, u = calcstDev(right_subDataSet)
            #stDev += right_prob * S
            stDev += S

            if (stDev < bestStDev):
                bestStDev = stDev
                bestFeature = i
                bestBinarySplit = (left, right)
    return bestFeature, bestBinarySplit, bestStDev

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(
            classCount.iteritems(), key=operator.itemgetter(1),
            reverse=True)
    return sortedClassCount[0][0]

def createTree(dataset, labels, originalS):
    classList = [example[-1] for example in dataset]
    #　所有实例属于同一类
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #　TODO
    if len(dataset) == 1:
        return majorityCnt(classList)
    bestFeat, bestBinarySplit, bestStDev = chooseBestFeatrueToSplit(dataset)
    if bestStDev < 0.05 * originalS:
        return 1.0 * np.mean(classList)
    bestFeatLabel = labels[bestFeat]
    if bestFeat == -1:
        return majorityCnt(classList)
    myTree = {bestFeatLabel: {}}
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = list(set(featValues))
    for value in bestBinarySplit:
        subLabels = labels[:]
        if len(value) < 2:
            del(subLabels[bestFeat])
        myTree[bestFeatLabel][value] = createTree(
                splitDataSet(dataset, bestFeat, value), subLabels, originalS)
    return myTree

def calcstDev(dataset):
    classList = [example[-1] for example in dataset]
    mean = np.mean(classList)
    std = np.std(classList)
    return std, mean

def main():
    filename = "temp"
    dataset, label = [], []
    with open(filename) as f:
        for line in f.readlines():
            fields = line.strip('\n').split(',')
            t = fields[:-1]
            t.append(int(fields[-1]))
            dataset.append(t)
    labels = ['outlook', 'temperature', 'humidity', 'windy']

    originalS, u = calcstDev(dataset)
    tree = createTree(dataset, labels, originalS)
    print(tree)


if __name__ == '__main__':
    main()

from __future__ import division
import csv
import random
import math


def openFile(filename):
    lines = csv.reader(open(filename))
    data = list(lines)
    count = 0
    for i in data:
        data[count] = [int(m) for m in i]
        count +=1
    return data

def splitData(data,splitRatio):
    trainSize = int(len(data)*splitRatio)
    trainData = []
    testData = list(data)
    for i in range(trainSize):
        index = random.randrange(len(testData))
        trainData.append(testData.pop(index))
    return trainData, testData


def dataClassification(data):
    classifier = {}
    for i in range(len(data)):
        vector = data[i]
        if vector[-1] not in classifier:
            classifier[vector[-1]] = []
        classifier[vector[-1]].append(vector)
    return classifier


def mean(data):
    return sum(data)/float(len(data))


def variance(data):
    avg = mean(data)
    return sum([pow(x - avg,2) for x in data])/float(len(data)-1)

def summation(data):
    distribution = [(mean(x), variance(x)) for x in zip(*data)]
    del distribution[-1]
    return distribution

def summationByClass(data):
    summaries = {}
    classifier = dataClassification(data)
    for classValue, instances in classifier.items():
        summaries[classValue] = summation(instances)
    return summaries

def gauss(x, m, v):
    return (1.0/float(math.sqrt(2*math.pi*v)))*math.exp(-(math.pow(x-m, 2)/float(2*v)))


def classProbabilities(data,inputVector):
    probabilities = {}
    for classValue, instances in data.items():
        probabilities[classValue] = 1
        for i in range(len(instances)):
            m, v = instances[i]
            x = inputVector[i]
            probabilities[classValue] *= gauss(x, m, v)
    return probabilities


def predict(data, inputVector):
    probabilities = classProbabilities(data, inputVector)
    bestProb, bestLabel = -1, None
    for classVal,instances in probabilities.items():
        if bestLabel is None or instances > bestProb:
            bestProb = instances
            bestLabel = classVal
    return bestLabel


def getPreds(data, testData):
    probabilities = []
    for i in testData:
        probabilities.append(predict(data, i))
    return probabilities


def getAccuracy(data,testData):
    probabilities = getPreds(data, testData)
    count = 0
    for i in range(len(probabilities)):
        if probabilities[i] == testData[i][-1]:
            count += 1
    return count*100/float(len(testData))


data = openFile("Lung Cancer Data.csv")

trainData, testData = splitData(data, 0.67)

summaries = summationByClass(trainData)

print("According to given data the accuracy of algorithm is {0}".format(getAccuracy(summaries,testData)))

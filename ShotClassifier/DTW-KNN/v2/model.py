from operator import le
from pickletools import read_uint1
from statistics import mean
from unicodedata import normalize
import csv
import numpy as np
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import glob
import sys
import os
import csv
# from fastdtw import fastdtw
# from scipy.spatial.distance import euclidean
import time
from dtaidistance import dtw
import array

def getshottype(filename):
    file = open(filename)
    file = csv.reader(file)
    count = 0
    for row in file:
        if(count == 1):
            typ = row[11]
        count=count+1
    return typ

def getfiledata(filename):
    dataset = np.loadtxt(filename, delimiter =",", dtype = float, skiprows=1, usecols = range(1, 11)).T
    dataset = dataset.tolist()
    return dataset

def fillDataSet(ds):  # function to fill the data set
    path = "../../../data"
    all_files = glob.glob(path + "/*.csv")
    for filename in all_files:
        filedata = getfiledata(filename)
        typ = getshottype(filename)
        tempdata = [filedata, int(typ)]
        ds.append(tempdata)


def getTestData(fn):  # function to get the test data
    # path = "../test/p1-s1-v1.csv"
    path = "../../../test/" + fn
    dataset = np.loadtxt(path, delimiter =",", dtype = float, skiprows=1, usecols = range(1, 11)).T
    dataset = dataset.tolist()
    return [dataset,getshottype(path)]


# function to normalize the array to zero mean and unit standard deviation
def normalizeArray(ar, mean, var):
    ar = (ar - mean) / var
    return ar

def getcostarr(ds, td):
    costarr = {}
    # print(np.array(td).shape)
    # print(len(ds))
    for i in range(len(ds)):
        features = ds[i][0]
        cost = 0
        for j in range(10):
            # print(features[j])
            distance = dtw.distance_fast(array.array('d', features[j]), array.array('d', td[j]), use_pruning = True)
            # print(distance)
            cost+=distance
        costarr[i] = cost
    return costarr

    

def knn(path , kVal):


    # st = time.time()
    dataSet = []  # [ [ [n X l Matrix] , type] , [[n X l Matrix] , type] , ... r times] : [n X l Matrix] = [[feature0 array] , [feature1 array] , [feature2 array] , ...[featurel array]]
    fillDataSet(dataSet)
    # print(dataSet)
    testData = getTestData(path)[0]# [m X l Matrix]
    testType = getTestData(path)[1]

    costs= getcostarr(dataSet, testData)
    # print(comp)
    # tempet = time.time()
    # print("cost time: " + str(tempet - st))

    costSorted = {k: v for k, v in sorted(costs.items(), key=lambda item: item[1])}
    # print(costSorted)

    k = kVal  # number of k neighbors to be considered
    z = 0
    indexes = []
    for i in costSorted:
        if z == k:
            break
        indexes.append(i)

        z+=1

    # print(indexes)

    neigbours = {1:0,2:0,3:0,4:0}
    for i in indexes:
        neigbours[dataSet[i][1]] += 1


    # print(neigbours)
    neigboursSorted = {k: v for k, v in sorted(neigbours.items(), key=lambda item: item[1])}
    # print(neigboursSorted)
    b = 0
    # print(testType)
    result = 0
    for i in neigboursSorted:
        if b == 3:
            ty = i
            if str(i) == str(testType):
                result = 1
        b+= 1

    # et = time.time()
    # print("hewwo")
    # print(result)
    # return result
    return ty , result
    # print(et-st)

# def isCorrect(path, kVal):
    # if str(knn(path,kVal)) == str()

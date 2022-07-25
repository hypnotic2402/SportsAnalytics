from operator import le
from statistics import mean
from unicodedata import normalize
import csv
import numpy as np
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import glob

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
    dataset = np.loadtxt(filename, delimiter =",", dtype = float, skiprows=1, usecols = range(1, 11))
    dataset = dataset.tolist()
    return dataset

def fillDataSet(ds):  # function to fill the data set
    path = "../data"
    all_files = glob.glob(path + "/*.csv")
    for filename in all_files:
        filedata = getfiledata(filename)
        typ = getshottype(filename)
        tempdata = [filedata, int(typ)]
        ds.append(tempdata)


def getTestData():  # function to get the test data
    path = "../test/p1-s1-v1.csv"
    dataset = np.loadtxt(path, delimiter =",", dtype = float, skiprows=1, usecols = range(1, 11))
    dataset = dataset.tolist()
    return dataset


# function to normalize the array to zero mean and unit standard deviation
def normalizeArray(ar, mean, var):
    ar = (ar - mean) / var
    return ar


def generateDistanceMatrix(td, ds, n):

    m = len(td)
    # print(m)
    # print(td[0])
    # print(m)
    distMat = np.zeros([len(ds), n, m])
    y = 0

    for x in ds:  # r such iterations
        multiDimSignal = x[0]
        l = len(multiDimSignal)
        n1 = len(multiDimSignal[0])

        meanVar = []
        # for i in range(l):
        #     meanVar.append([0,0])

        # for i in range(l):
        #     arr = np.array(multiDimSignal[i])
        #     tempVar = [np.mean(arr), np.std(arr)]
        #     meanVar.append(tempVar)

        # temp = np.empty([l, n1])
        # for i in range(l):
        #     temp = np.append(temp, (np.array(multiDimSignal[i]) - meanVar[i][0]) / meanVar[i][1])

        # print(len(temp))
        # temp now holds normalized vlues of each feature's vector

        # Smoothen with gaussian if neccessary later
        # print(l)
        # print(m)
        # print(n1)
        # print(len(td))
        # print(distMat[0])
        # print(td)
        # print(multiDimSignal)
        # for i in range(n1):
        #     for j in range(m):
        #         for k in range(l):
        #             # print(multiDimSignal[i])
        #             distMat[y, i, j] += abs(multiDimSignal[i][k] - td[j][k])
                    
                    # distMat[y, i, j] += abs(temp[i,k] - td[j][k])

        # for frameTrain in multiDimSignal:
        #     for frameTest in td:
        #         for nfeatures in range(10):

        for i in range(52):
            # print(y)
            for j in range(len(td)):
                for nfeatures in range(10):
                    print("i = " + str(i) + ", j = " + str(j) + ",k = " + str(nfeatures))
                    distMat[y, i , j] += abs(multiDimSignal[i][nfeatures] - td[j][nfeatures])



        y += 1
    
    return distMat


def calcDTWdistance(distMat , l ) :  # function to retrun the DTW allignment cost
    cd = []
    for cdIndex in range(l):
        N, M = distMat[cdIndex].shape
        cost_mat = np.zeros((N + 1, M + 1))
        for i in range(1, N + 1):
            cost_mat[i, 0] = np.inf
        for i in range(1, M + 1):
            cost_mat[0, i] = np.inf

        traceback_mat = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                penalty = [
                    cost_mat[i, j],      # match (0)
                    cost_mat[i, j + 1],  # insertion (1)
                    cost_mat[i + 1, j]]  # deletion (2)
                i_penalty = np.argmin(penalty)
                cost_mat[i + 1, j + 1] = distMat[cdIndex,i, j] + penalty[i_penalty]
                traceback_mat[i, j] = i_penalty

        # Traceback from bottom right
        i = N - 1
        j = M - 1
        path = [(i, j)]
        while i > 0 or j > 0:
            tb_type = traceback_mat[i, j]
            if tb_type == 0:
                # Match
                i = i - 1
                j = j - 1
            elif tb_type == 1:
                # Insertion
                i = i - 1
            elif tb_type == 2:
                # Deletion
                j = j - 1
            path.append((i, j))

        # Strip infinity edges from cost_mat before returning
        cost_mat = cost_mat[1:, 1:]
        # return (cost_mat)
        cd.append(cost_mat)
    return cd

# l = number of features
# n = length of each feature's array
# m = length of test data array
# r = number of labled sets

# type = 0 , 1 , 2 or 3


dataSet = []  # [ [ [n X l Matrix] , type] , [[n X l Matrix] , type] , ... r times] : [n X l Matrix] = [[feature0 array] , [feature1 array] , [feature2 array] , ...[featurel array]]
fillDataSet(dataSet)
# print(dataSet)
testData = getTestData()# [m X l Matrix]
# print(testData)

l = len(dataSet)
n = len(dataSet[0][0])
# print(dataSet[0])
# print("n = " +str(n))


distMat = generateDistanceMatrix(testData, dataSet, n)

costData = calcDTWdistance(distMat, l ) # r long array


# for i in range(l):
#     costData.append(calcDTWdistance([distMat[i] , i]))


print(costData)

k = 5  # number of k neighbors to be considered




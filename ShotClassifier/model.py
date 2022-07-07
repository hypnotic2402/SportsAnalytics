from unicodedata import normalize
import numpy as np
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt

def fillDataSet(ds):  # function to fill the data set
    pass


def getTestData(td):  # function to get the test data
    pass


# function to normalize the array to zero mean and unit standard deviation
def normalizeArray(ar, mean, var):
    ar = (ar - mean) / var
    return ar


def generateDistanceMatrix(td, ds):

    m = td[0].length()
    distMat = np.zeros([ds.length(), n, m])
    y = 0

    for x in ds:  # r such iterations
        multiDimSignal = x[0]
        l = multiDimSignal.length()
        n = multiDimSignal[0].length()

        meanVar = []
        # for i in range(l):
        #     meanVar.append([0,0])

        for i in multiDimSignal.length():
            arr = np.array(multiDimSignal[i])
            meanVar.append([np.mean(arr), np.std(arr)])

        temp = np.empty([l, n])
        for i in range(l):
            temp.append(
                (np.array(multiDimSignal[i]) - meanVar[i, 0]) / meanVar[i, 1])

        # temp now holds normalized vlues of each feature's vector

        # Smoothen with gaussian if neccessary later

        for i in range(n):
            for j in range(m):
                for k in range(l):
                    distMat[y, i, j] += abs(multiDimSignal[i, k] - td[j, k])

        y += 1

    return distMat


def calcDTWdistance(distMat):  # function to retrun the DTW allignment cost
    N, M = distMat.shape
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
            cost_mat[i + 1, j + 1] = distMat[i, j] + penalty[i_penalty]
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
    return (cost_mat)

# l = number of features
# n = length of each feature's array
# m = length of test data array
# r = number of labled sets

# type = 0 , 1 , 2 or 3


dataSet = []  # [ [ [n X l Matrix] , type] , [[n X l Matrix] , type] , ... r times] : [n X l Matrix] = [[feature0 array] , [feature1 array] , [feature2 array] , ...[featurel array]]
fillDataSet(dataSet)

testData = []  # [m X l Matrix]
getTestData(testData)

l = dataSet.length()
n = dataSet[0].length()


distMat = generateDistanceMatrix(testData, dataSet)

costData = []  # r long array


for i in dataSet.length():
    costData.append(calcDTWdistance([distMat[i] , i]))
    


k = 5  # number of k neighbors to be considered




from unicodedata import normalize
import numpy as np

def fillDataSet(ds): #function to fill the data set
    pass

def getTestData(td): #function to get the test data
    pass

def normalizeArray(ar , mean , var): #function to normalize the array to zero mean and unit standard deviation
    ar = (ar - mean) / var
    return ar

def generateDistanceMatrix(distMat , td , ds):
    
    for x in ds: # r such iterations
        multiDimSignal = x[0]
        l = multiDimSignal.length()
        n = multiDimSignal[0].length()

        meanVar = []
        # for i in range(l):
        #     meanVar.append([0,0])

        for i in multiDimSignal.length():
            arr = np.array(multiDimSignal[i])
            meanVar.append([np.mean(arr) , np.std(arr)])

        temp = np.empty([l , n])
        for i in range(l):
            temp.append((np.array(multiDimSignal[i]) - meanVar[i,0]) / meanVar[i,1]) 

        # temp now holds normalized vlues of each feature's vector

        # Smoothen with gaussian if neccessary later



        




        
            
    

def calcDTWdistance(distMat): #function to retrun the DTW allignment cost
    return 0

# l = number of features
# n = length of each feature's array
# m = length of test data array
# r = number of labled sets

# type = 0 , 1 , 2 or 3



dataSet = [] # [ [ [n X l Matrix] , type] , [[n X l Matrix] , type] , ... r times] : [n X l Matrix] = [[feature0 array] , [feature1 array] , [feature2 array] , ...[featurel array]]
fillDataSet(dataSet)

testData = [] # [m X l Matrix]
getTestData(testData)

l = dataSet.length()
n = dataSet[0].length()


distMat = np.zeros([]) # [ [n X m Matrix] , [n X m Matrix] , [n X m Matrix] , ... r times]
generateDistanceMatrix(distMat, testData , dataSet)

costData = [] # r long array

for i in dataSet.length():
    costData.append(calcDTWdistance(distMat[i]))

costData.sort()

k = 5 # number of k neighbors to be considered
final = [0 , 0 , 0 , 0]







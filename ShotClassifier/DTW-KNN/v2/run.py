import os

from model import *

if __name__ == '__main__':
    testPath = str(input("Test File Name : "))
    k = int(input("K = "))
    stype , corr = knn(testPath , k)
    print("Shot Type : " + str(stype))

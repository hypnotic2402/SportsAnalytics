from unittest import skip
import numpy as np

dataset = np.loadtxt('../data/p1-s1-v1.csv', delimiter =",", dtype = float, skiprows=1, usecols = range(1, 11))
print(dataset)
from re import T
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('digit-recognizer/train.csv')

data.head()

data = np.array(data)
m,n = data.shape
np.random.shuffle(data)

data_dev = data[0:10000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

print(X_train[0].shape)
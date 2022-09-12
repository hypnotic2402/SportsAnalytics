import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import *

from keras.models import Model
from keras.layers import Input, PReLU, Dense, LSTM, multiply, concatenate, Activation
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from keras.callbacks import ModelCheckpoint
from tensorflow import keras
from tensorflow.keras import layers



def givePred(data , model_path):
    model = keras.models.load_model(model_path)
    x = data
    # x = x.to_numpy()
    x = np.expand_dims(x , axis=0)
    print(x.shape)

    
    return (np.argmax(model.predict(x , verbose = 0)) + 1)


# print(givePred('testMatrices/xTest-0-2.csv', 'badminton2.h5'))
    




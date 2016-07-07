import numpy as np
import pandas as pd
from keras.utils import np_utils


training = pd.read_csv('/Users/hanzhang/Google Drive/data/mnist/train.csv')

# split training labels and pre-process them
training_targets = training.ix[:,0].values.astype('int32')
print training_targets[0]

training_targets = np_utils.to_categorical(training_targets)
print training_targets[0][1]
# split training inputs
training_inputs = (training.ix[:,1:].values).astype('float32')

# read testing data
testing_inputs = (pd.read_csv('/Users/hanzhang/Google Drive/data/mnist/test.csv').values).astype('float32')

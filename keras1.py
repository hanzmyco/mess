from keras.models import Sequential
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD
from keras.datasets import cifar100

model=Sequential()
# conv1
model.add(Convolution2D(96,11,11,border_mode='same',input_shape=(3,224,224),subsample=(4,4)))
model.add(Activation('relu'))
model.add(BatchNormalization(mode=1))
model.add(MaxPooling2D(pool_size=(2,2)))
print model.output_shape
#conv2
model.add(Convolution2D(256,5,5,border_mode='same',input_shape=(96,28,28)))
model.add(Activation('relu'))
model.add(BatchNormalization(mode=1))
model.add(MaxPooling2D(pool_size=(2,2)))
#conv3
model.add(Convolution2D(384,3,3,border_mode='same',input_shape=(256,14,14)))
model.add(Activation('relu'))
print model.output_shape
#conv4
model.add(Convolution2D(384,3,3,border_mode='same',input_shape=(384,14,14)))
model.add(Activation('relu'))
print model.output_shape
#conv5
model.add(Convolution2D(256,3,3,border_mode='same',input_shape=(384,14,14)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
print model.output_shape
# full connected layer6
model.add(Flatten())
model.add(Dense(output_dim=4096))
model.add(Activation('relu'))
model.add(Dropout(0.25))
print model.output_shape
#full connected layer7
model.add(Dense(output_dim=4096))
model.add(Activation('relu'))
model.add(Dropout(0.25))
print model.output_shape
#software layer7
model.add(Dense(output_dim=1000))
model.add(Activation('softmax'))
print model.output_shape
# setting sgd optimizer parameters
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

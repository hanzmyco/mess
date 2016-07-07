from keras.models import Sequential
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD
from keras.datasets import cifar100
from keras.utils import np_utils


model=Sequential()
# conv1
model.add(Convolution2D(48,2,2,border_mode='same',input_shape=(3,32,32),subsample=(2,2)))
model.add(Activation('relu'))
model.add(BatchNormalization(mode=1))
model.add(MaxPooling2D(pool_size=(2,2)))
print model.output_shape

#conv2
'''
model.add(Convolution2D(256,2,2,border_mode='same',input_shape=(96,8,8)))
model.add(Activation('relu'))
model.add(BatchNormalization(mode=1))
model.add(MaxPooling2D(pool_size=(2,2)))
print model.output_shape
'''

#conv3
'''
model.add(Convolution2D(384,2,2,border_mode='same',input_shape=(256,4,4)))
model.add(Activation('relu'))
print model.output_shape
'''

#conv4

model.add(Convolution2D(384,2,2,border_mode='same',input_shape=(48,8,8)))
model.add(Activation('relu'))
print model.output_shape

#conv5
'''
model.add(Convolution2D(256,2,2,border_mode='same',input_shape=(384,4,4)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
print model.output_shape
'''
model.add(Flatten())


# full connected layer6
model.add(Dense(output_dim=512))
model.add(Activation('relu'))
model.add(Dropout(0.25))
print model.output_shape
'''
#full connected layer7
model.add(Dense(output_dim=512))
model.add(Activation('relu'))
model.add(Dropout(0.25))
print model.output_shape
'''
#software layer8
model.add(Dense(output_dim=100))
model.add(Activation('softmax'))
print model.output_shape

# setting sgd optimizer parameters
sgd = SGD(lr=0.001, decay=1e-10, momentum=0.9, nesterov=True,clipnorm=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
y_train= np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
print y_test[0]

train_inputs=X_train.reshape(X_train.shape[0],3,32,32)
test_inputs=X_test.reshape(X_test.shape[0],3,32,32)
print train_inputs[0]
model.fit(train_inputs,y_train,nb_epoch=10, batch_size=1000, validation_split=0.1, show_accuracy=True)
print("Generating predections")
preds = model.predict_classes(test_inputs, verbose=0)
for i in preds:
    print i
print y_test

json_string = model.to_json()
open('my_model_architecture2.json', 'w').write(json_string)
model.save_weights('my_model_weights2.h5')

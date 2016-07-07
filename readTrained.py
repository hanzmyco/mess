from __future__ import division
from keras.models import Sequential,model_from_json
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD
from keras.datasets import cifar100
model = model_from_json(open('my_model_architecture1.json').read())
model.load_weights('my_model_weights1.h5')
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
test_inputs=X_test.reshape(X_test.shape[0],3,32,32)
preds = model.predict_classes(test_inputs, verbose=0)

correct=0
for i in xrange(0,len(preds)):
    print y_test[i][0],
    print '  ',
    print preds[i]
    if y_test[i][0]==preds[i]:
        correct+=1
print correct/len(preds)

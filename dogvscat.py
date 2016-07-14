from keras.models import Sequential
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 224, 224
prefix='/Users/hanzhang/Google Drive/data/dogvscat'

train_data_dir = prefix+'/train'
validation_data_dir = prefix+'/validate'
nb_train_samples = 2000
nb_validation_samples = 800
nb_epoch = 50
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
model.add(Dense(output_dim=1))
model.add(Activation('softmax'))
print model.output_shape
# setting sgd optimizer parameters
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')
model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)
json_string = model.to_json()
open('dogcat_model_architecture.json', 'w').write(json_string)
model.save_weights('dogcat_model_weights.h5')

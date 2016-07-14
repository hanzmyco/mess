from __future__ import division
from keras.models import Sequential,model_from_json
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
def predict_image(input1,target_size=(150,150)):
    img=load_img(input1,target_size=(150,150))
    x=img_to_array(img)
    x=x.reshape((1,)+x.shape)
    return model.predict_classes(x)

model = model_from_json(open('dogcat_model_architecture.json').read())
model.load_weights('dogcat_model_weights.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

img_width, img_height = 150, 150
prefix='/Users/hanzhang/Google Drive/data/dogvscat'
test_data_dir=prefix+'/test'
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        #batch_size=32,
        class_mode='binary')
#print (model.predict_generator(test_generator,3))

jk='/dog/1.jpg'
dog='/dog/dog.0.jpg'
cat='/cat/cat.0.jpg'
validation_data_dir = prefix+'/validate'
print predict_image(test_data_dir+jk,(150,150))


for i in xrange(1,8):
    cat_name='/dog/'+str(i)+'.jpg'
    print predict_image(test_data_dir+cat_name,(150,150))

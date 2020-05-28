from keras import models
from keras import layers

from glob import glob
train_dir = "C:\\Users\\kkodw\\Desktop\\AIProject\\aiproject_deeplearning\\New_Data_Set\\train"
test_dir = "C:\\Users\\kkodw\\Desktop\\AIProject\\aiproject_deeplearning\\New_Data_Set\\test"


from keras.applications import VGG19
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

conv_base = VGG19(weights='imagenet',
include_top=False,
input_shape=(150, 150, 3))
from keras import models
from keras import layers
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

glob(train_dir)

model.compile(loss='categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=32,
                                                 class_mode='categorical')
test_set = test_datagen.flow_from_directory(test_dir,target_size=(150,150),batch_size=32
                                     ,class_mode='categorical')

model.fit_generator(training_set,validation_data=test_set,epochs=5,steps_per_epoch=20,
                    validation_steps=5)


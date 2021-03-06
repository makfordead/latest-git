from keras.applications import VGG16
import cv2
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

conv_base = VGG16(weights='imagenet',
include_top=False,
input_shape=(150, 150, 3))
from keras import models
from keras import layers
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')
from google.colab.patches import cv2_imshow
test_datagen = ImageDataGenerator()

train_dir = "/content/AI_project_deep/dataset/train"
validation_dir = "/content/AI_project_deep/dataset/validation"

train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(150, 150),
batch_size=20,
class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
validation_dir,
target_size=(150, 150),
batch_size=20,
class_mode='categorical')


model.compile(loss='categorical_crossentropy',
optimizer=optimizers.RMSprop(lr=2e-5),
metrics=['acc'])


history = model.fit_generator(
train_generator,
steps_per_epoch=30,
epochs=1,
validation_data=validation_generator,
validation_steps=50)




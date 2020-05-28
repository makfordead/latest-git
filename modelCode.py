from keras import layers
from keras import models
import os, shutil
import cv2
from glob import glob
import matplotlib.pyplot as plt
dataset_directory = "/home/kapilkodwani/Music/AI_DEEP_Learning_Project/dataset"
dataset_directory = os.path.join(dataset_directory,"train")


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(500, 500, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

from keras import optimizers
model.compile(loss="categorical_crossentropy",optimizer=optimizers.RMSprop(learning_rate=0.001),metrics=['acc'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(dataset_directory,target_size=(500, 500),batch_size=2,class_mode='categorical')

y_true_labels = train_generator.classes


history = model.fit_generator(train_generator,steps_per_epoch=10,epochs=5)

labels = os.listdir("/home/kapilkodwani/Music/AI_DEEP_Learning_Project/dataset/train/")
print(sorted(labels))


img = cv2.imread("/home/kapilkodwani/Music/AI_DEEP_Learning_Project/dataset/test/fasiha/f1.jpg")
img = cv2.resize(img,(500,500),interpolation=cv2.INTER_AREA)
cv2.imshow('image',img)
cv2.waitKey(0)

img = img.reshape((1,)+img.shape)
result = model.predict(img)
result_class = model.predict_classes(x=img)
print(result_class)
print(result)

from keras import models
import cv2
from keras_preprocessing.image import ImageDataGenerator

model = models.load_model('model.h5')

img = cv2.imread("C:\\Users\\kkodw\\Desktop\\AIProject\\aiproject_deeplearning\\dataset\\train\\fasiha\\ahmed1.jpg")
img = cv2.resize(img,(150,150),interpolation=cv2.INTER_AREA)
cv2.imshow('image',img)
train_dir = "C:\\Users\\kkodw\\Desktop\\AIProject\\aiproject_deeplearning\\dataset\\train"

train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')
img = img.reshape((1,)+img.shape)
result = model.predict(img)
cv2.waitKey(0)
train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(150, 150),
batch_size=20,
class_mode='categorical')
print(train_generator.class_indices)

result_class = model.predict_classes(x=img)
print(result_class)
print(result)
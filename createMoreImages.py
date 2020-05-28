from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import cv2
from face_crop import extractFace
datagen = ImageDataGenerator(
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
rescale=1./255,
brightness_range=[0.2,1.0]
,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
)
pic = cv2.imread("C:\\Users\\kkodw\\Desktop\\AIProject\\aiproject_deeplearning\\dataset\\validation\\ahmed\\ahmed.jpg")
#remove
#pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
cv2.imshow('image',pic)
cv2.waitKey(0)
pic = extractFace(pic)
cv2.imshow('image',pic)
cv2.waitKey(0)
#cv2.waitKey(0)
pic_array = img_to_array(pic)
pic_array = pic_array.reshape((1,)+ pic_array.shape)
print(pic_array.shape)
count = 0
for batch in datagen.flow(pic_array,batch_size=5,save_to_dir="C:\\Users\\kkodw\\Desktop\\AIProject\\aiproject_deeplearning\\dataset\\validation\\ahmed\\",save_prefix="ahmed"):
    count += 1
    if count == 40:
        break

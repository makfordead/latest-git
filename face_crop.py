import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def extractFace(img):
    face = face_classifier.detectMultiScale(img,1.3,5)
    cropped_face = None
    for (x,y,w,h) in face:
        x = x-10
        y = y-10
        cropped_face = img[y:y+h+50,x:x+w+50]
    return cropped_face
pic = cv2.imread("C:\\Users\\kkodw\\Desktop\\AIProject\\aiproject_deeplearning\\dataset\\test\\manisha\\manisha.jpeg")
#pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
pic = extractFace(pic)

cv2.imwrite("C:\\Users\\kkodw\\Desktop\\AIProject\\aiproject_deeplearning\\dataset\\test\\manisha\\manisha1.jpeg",pic)

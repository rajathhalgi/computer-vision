import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('photos/group_1.jpg')
cv.imshow('person',img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=6)
# if there is noise then just try to increase the min neighbours
# if there are many people in the picture, try min neighbours as 1 or 2. but the tradeoff is noise
# the least min neighbours is prone to noise
print(f'Number of faces found = {len(faces_rect)}')

for(x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h),(0,255,0), thickness=2)

cv.imshow('Detected faces', img)
cv.waitKey(0)
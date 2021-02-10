import cv2 as cv
import numpy as np
img = cv.imread('photos/cats.jpg')
cv.imshow('Cats', img)

# Masking allows us to focus on certain parts of an image that we would like to focus on
blank = np.zeros(img.shape[:2], dtype='uint8')
cv.imshow('Blank', blank)

circle = cv.circle(blank.copy(),(img.shape[1]//2, img.shape[0]//2), 100,255,-1)
cv.imshow('Mask', circle)

rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255,-1)
cv.imshow('rectangle', rectangle)

wierd = cv.bitwise_and(circle,rectangle)
cv.imshow('wierd',wierd)

masked = cv.bitwise_and(img,img,mask=wierd )
cv.imshow('Masked', masked)

cv.waitKey(0)
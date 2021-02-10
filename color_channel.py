import cv2 as cv
import numpy as np

# spliting an image into respective three colors
img = cv.imread('photos/boston.jpeg')
cv.imshow('Boston', img)
blank = np.zeros(img.shape[:2], dtype='uint8')

b,g,r = cv.split(img)

blue = cv.merge([b,blank,blank]) 
green = cv.merge([blank, g, blank])
red = cv.merge([blank,blank,r])

cv.imshow('Blue', blue)
cv.imshow('Green', green)
cv.imshow('Red', red)

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

# Regions where the intensity is lighter this means far more concentration of pixel values, Dark means no pixels

merged = cv.merge([b,g,r])
cv.imshow('Merged', merged)

cv.waitKey(0)
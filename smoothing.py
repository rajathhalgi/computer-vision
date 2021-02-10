import cv2 as cv
import numpy as np


img = cv.imread('photos/cats.jpg')
cv.imshow('Cats', img)

# averaging
average = cv.blur(img, (3,3), )
cv.imshow('Average blur', average)

# gausian blur
gauss = cv.GaussianBlur(img, (3,3), 0)
cv.imshow('GBLUR', gauss)

# median Blur(more effective in removing noise in the image)
median = cv.medianBlur(img, 3)
cv.imshow('Median', median)

#bilateral blur (most effective)
bilateral = cv.bilateralFilter(img, 10, 35, 25)
cv.imshow('bilateral', bilateral)

cv.waitKey(0) 
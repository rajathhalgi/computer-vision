import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Thresholding is binarization of an image
# In general, we take an image and convert it into a binary image
# i.e where pixels are either 0 (black) and 255 (white)
# set up a threshold value, if the intensity of the pixel is less than threshold than set the value to 0 and if more then 255

img = cv.imread('photos/cats.jpg')
cv.imshow('Cats', img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

#Simple Thresholding:
threshold, thresh = cv.threshold(gray, 150, 255,cv.THRESH_BINARY)
cv.imshow('Simple Thresholded',thresh)

# cv.threshold function returns threshold and thresh(binarized image)
# gray scaled image has to be passed in as src
# Threshold value
# max value: if that pixel value is greater than 150 then it is set to 255.7
# thresh type is cv.THRESH_BINARY

threshold, thresh_inv = cv.threshold(gray, 150, 255,cv.THRESH_BINARY_INV)
cv.imshow('Simple Thresholded Inverse',thresh_inv)

# Adaptive thresholding
adaptive_thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,3)
cv.imshow('Adaptive',adaptive_thresh)

# cv.ADAPTIVE_THRESH_MEAN_C: mean of neighbourhood pixels|| can also use cv.ADAPTIVE_THRESH_GAUSSIAN_C
# block size: neighborhood size of the kernel size which opencv needs to use to compute the mean for optimal threshold value
# C : an integer that is subtracted from the mean to fine tune the threshold

cv.waitKey(0)
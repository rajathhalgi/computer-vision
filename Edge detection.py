import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('photos/boston.jpeg')
cv.imshow('boston', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

# There are two other ways to compute edges in an image other than canny
# 1: Laplacian
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian',lap)

# 2: Sobel : computes gradients in x and y direction
sobelx = cv.Sobel(gray, cv.CV_64F, 1,0)
sobely = cv.Sobel(gray, cv.CV_64F, 0,1)

combined_sobel = cv.bitwise_or(sobelx,sobely)
cv.imshow('Combined', combined_sobel)
cv.imshow('Sobel_X',sobelx)
cv.imshow('Sobel_Y', sobely)

# 3: canny
canny = cv.Canny(gray,150,175)
cv.imshow('canny',canny)
cv.waitKey(0)
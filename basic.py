import cv2 as cv
import numpy as np

img = cv.imread('photos/boston.jpeg')
cv.imshow('boston', img)

# lets convert this into grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# blurring an image(Removes some of the noise in the image)
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

# Edge Cascade
canny = cv.Canny(blur, 125,175)
cv.imshow('canny edges', canny)

# dilating the images
dilated = cv.dilate(canny, (7,7), iterations=3)
cv.imshow('Dilated', dilated)

# Eroding
eroded = cv.erode(dilated, (3,3), iterations=1)
cv.imshow('Eroded', eroded)

# Resize and crop an image
resized = cv.resize(img,(500,500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

# cropping
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)
cv.waitKey(0)



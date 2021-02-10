import cv2 as cv
import matplotlib.pyplot as plt
# how to switch between color spaces in open cv
# color spaces are system representing an array of pixel colors
img = cv.imread('photos/boston.jpeg')
cv.imshow('Boston', img)

# plt.imshow(img)
# plt.show()

# BGR to grayscale
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)

# BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) 
cv.imshow('hsv',hsv)

#BGR to LAB
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB', lab)

#BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('rbg', rgb)

# we can convert the other way around now
# problem is that we cannot convert grayscale to hsv directly. We have to convert greysccale into bgr first

# HSV to bgr
hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.imshow('hsv to bgr', hsv_bgr)
plt.imshow(rgb)
plt.show()


cv.waitKey(0)
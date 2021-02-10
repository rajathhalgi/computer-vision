import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img = cv.imread('photos/cats.jpg')
cv.imshow('Cats', img)

blank = np.zeros(img.shape[:2], dtype='uint8')


# Histogram Computing
# it allows us to visualize the pixel intensity distribution in an image

# HC for grayscale
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

#circle = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100,255,-1)

# mask = cv.bitwise_and(gray,gray, mask=circle)
# cv.imshow('Mask', mask)

# gray_hist = cv.calcHist([gray],[0],mask, [256],[0,256])
# attributes are:
# images are the list of images
# channels are index of channel  we want to compute a hystogram for
# mask are histogram for a specific portion of an image
# histsize is the number of bins that we want to use for computing histogram
# Ranges are the all possible pixel values
# bins are the intervals of pixel intensities

# plt.figure()
# plt.title('Greyscale Histogram')
# plt.xlabel('Bins') 
# plt.ylabel('# of pixels')
# plt.plot(gray_hist)
# plt.xlim([0,256])
# plt.show()

# color histogram for rgb
mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100,255,-1)
masked = cv.bitwise_and(img,img, mask=mask)
cv.imshow('Mask', masked)

plt.figure()
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')

colors = ('b','g','r')
for i , col in enumerate(colors):
    hist = cv.calcHist([img],[i],mask,[256],[0,256])
    plt.plot(hist, color=col)
    plt.xlim([0,256])

plt.show()

cv.waitKey(0)
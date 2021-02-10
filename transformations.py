import cv2 as cv
import numpy as np
# image transformations
img = cv.imread('photos/boston.jpeg')
cv.imshow('Boston', img)


# 1 Translation = Shifting an image along the X and Y axis
def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img,transMat,dimensions)

# -X ---> Shifting the image Left    [0] is height and [1] is width
# -y ---> Shifting the image up
# X ---> Shifting the image right 
# y --->Shifting the image down

translated = translate(img,-100,-100)
cv.imshow('Translated',translated)

# 2 Rotation
def rotate(img, angle, rotPoint = None):
    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2,height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint,angle,1.0)
    dimensions = (width,height)

    return cv.warpAffine(img, rotMat,dimensions)

rotated = rotate(img,-45)
cv.imshow('Rotated', rotated)

#rotated_rotated = rotate(rotated, -45)
#cv.imshow('Rots', rotated_rotated)

#rot3 = rotate(rotated_rotated, -45)
#cv.imshow('Rot3',rot3)

# Resized
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

# Flipping an image
flip = cv.flip(img, 1)
# o is flipping the image vertically, 1 specifies flipping of the image horizontally over the y axis
# -1 flipping the image both vertically and horizontally
cv.imshow('Flip', flip)

# Cropping
cropped = img[200:400, 300:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)
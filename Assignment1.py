# 1. Split between BGR channels

import cv2 as cv
import numpy as np

#read img
original = cv.imread('rainbow.png')

#resize imgage
scale_percent = 40 # percent of original size
width = int(original.shape[1] * scale_percent / 100)
height = int(original.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv.resize(original, dim, interpolation = cv.INTER_AREA)

#split color
#(B,G,R) = cv.split(img)

#change middle pixel
#img[125, 150] = (255, 255, 255)
cv.imshow('Original', img)
#print(img.shape [:])

B = img[:,:,0]
G = img[:,:,1]
R = img[:,:,2]

#B, G, R = cv.split(img)

cv.imshow('Blue', B)
cv.imshow('Green', G)
cv.imshow('Red', R)

# 2. Merge BGR channels

#merge image
merge = cv.merge([B,G,R])
cv.imshow('Merge',merge)

cv.waitKey(0)
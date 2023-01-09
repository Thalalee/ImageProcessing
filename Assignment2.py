import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

os.system('cls')

#read img
original = cv.imread('rainbow.png')

#resize imgage
scale_percent = 40 # percent of original size
width = int(original.shape[1] * scale_percent / 100)
height = int(original.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv.resize(original, dim, interpolation = cv.INTER_AREA)

#----------------------------------------------------------
# Prob.1 convert RGB to grayscale

#convert to grayscale using opencv
gray1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# convert to grayscale from scratch. The channel order is BGR
gray2 = img[:, :, 2]*0.299 + img[:, :, 1]*0.587 + img[:, :, 0]*0.114
gray2 = gray2.astype(np.uint8)

cv.imshow('Original', img)
cv.imshow('grayscale from opencv', gray1)
cv.imshow('grayscale', gray2)

#=========================================================
# Prob.2 histogram

# Plot histogram from opencv
hist1 = cv.calcHist([gray1],[0],None,[256],[0,256])
plt.subplot(211)
plt.plot(hist1, color='b')
plt.title('Histogram from Opencv')

#---------------------------------------------------------

# get the row dimension of the image
row, col = gray1.shape[0], gray1.shape[1]
dim = row*col

# Create array to store values
h = [None] * dim
x = [None] * 256
k = 0

# Get intensity of each pixel and store in array h
for i in range(0, row):
    for j in range(0, col):
      h[k] = gray1[i, j]
      j += 1
      k += 1
    i += 1

# count number of each intensity to create a histogram
for l in range(0,256):
  x[l] = h.count(l)

plt.subplot(212)
plt.plot(x)
plt.title('Histogram from Opencv')
plt.show()
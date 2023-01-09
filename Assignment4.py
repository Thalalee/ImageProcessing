import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

os.system('cls')

#read img
img = plt.imread('Pic.jpg')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=0.8)
plt.subplot(141), plt.imshow(img), plt.title('Original')

#------------------------------------------------------------------------------

# 1. Construct codes for Contrast Stretching
# RGB to Gray
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
plt.subplot(142), plt.imshow(gray, cmap='gray'), plt.title('gray')

minI    = min(gray.min(axis=1))
maxI    = max(gray.max(axis=1))
minO    = 0
maxO    = 255

# Contrast stretching in gray scale
def ContrastStretching(gray) :
    cs = np.zeros(gray.shape)
    
    for i in range(len(gray)):
        for j in range(len(gray[0])):
            cs[i,j] = (gray[i,j]-minI)*(((maxO-minO)/(maxI-minI))+minO)

    return cs
cs = ContrastStretching(gray)
plt.subplot(143), plt.imshow(cs, cmap='gray'), plt.title('Contrast Stretching')

#--------------------------------------------------------------------------------------------

# 2. Construct the codes for Modified contrast streching
low = np.percentile(gray, 5)
high = np.percentile(gray, 95)

def ModifiedContrastStretching(input) :
    output = np.zeros(input.shape)
    
    for i in range(len(input)):
        for j in range(len(input[0])):
            output[i,j] = (input[i,j]-minI)*(((maxO-minO)/(maxI-minI))+minO)
        
    return output

percentile = np.where(gray > low, gray, minO)
percentile = np.where(gray < high, percentile, maxO)

mcs = ModifiedContrastStretching(percentile)

plt.subplot(144), plt.imshow(mcs, cmap='gray'), plt.title('Modified Contrast Stretching')
plt.show()

#-------------------------------------------------------------------------------

# 3. Plot histogram Histogram between gray image and modified contrast stretching

plt.xlim([0, 256])
histogram, bin_edges = np.histogram(gray, bins=256, range=(0, 256))
histogram_mcs = np.histogram(mcs, bins=256, range=(0, 256))
plt.plot(bin_edges[0:-1], histogram_mcs, histogram, histogram_mcs)
plt.show()

# --------------------------------------------------------------------------------

# 4. Thresholding from modified contrast streching

# read img
from imageio import imread
img1 = imread('https://thumbs.dreamstime.com/b/simple-flower-background-simple-flower-background-petals-summer-season-allergy-plant-152750248.jpg')
plt.subplot(421), plt.imshow(img1, cmap='gray'), plt.title('Original')

# RGB to Gray
gray1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
plt.subplot(422), plt.imshow(gray1, cmap='gray'), plt.title('gray')

# Modified contrast streching
t = np.percentile(gray1, 50)

thre = np.where(gray1 > t, gray1, minO)
thre = np.where(gray1 < t, thre, maxO)

plt.subplot(121), plt.imshow(gray1, cmap='gray'), plt.title('Original')
plt.subplot(122), plt.imshow(thre, cmap='gray'), plt.title('Thresholding')
plt.show()

# ----------------------------------------------------------------------

# 5. Histogram Equalization

import numpy as np

his_e = np.zeros(histogram.shape)
sum = 0
mul = 1

for i in range(histogram.shape[0]):
  sum = sum + histogram[i]
  his_e[i] = sum

his_e = his_e/his_e[255]

his_e = np.rint(his_e*np.max(gray))

plt.xlim([0, 256])
histogram_e, bin_edges = np.histogram(his_e, bins=256, range=(0, 256))
plt.plot(bin_edges[0:-1], histogram_e)
plt.show()

img_his_e = gray

for i in range(np.max(gray), -1, -1):
  img_his_e = np.where(img_his_e == i, his_e[i], img_his_e)

plt.subplot(121), plt.imshow(gray, cmap='gray'), plt.title('Original')
plt.subplot(122), plt.imshow(img_his_e, cmap='gray'), plt.title('Histogram equalization')
plt.show()

# ----------------------------------------------------------------------

# 6. Construct code for Matched Histogram Equalization

import matplotlib.pyplot as plt
from skimage import exposure
from skimage.exposure import match_histograms
import cv2
  
image = gray
reference = img_his_e
  
matched = match_histograms(image, reference , multichannel=True)

plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.subplot(132), plt.imshow(reference, cmap='gray'), plt.title('Histogram equalization')
plt.subplot(133), plt.imshow(matched, cmap='gray'), plt.title('Histogram equalization')
plt.tight_layout()
plt.show()

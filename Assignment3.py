# Convert RGB image to HSV, CMYK, and LAB by hand

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

os.system('cls')

#read img
img = cv.imread('Pic.jpg')

#-------------------------------------------------------------

#RGB to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

H = hsv[:,:,0]
S = hsv[:,:,1]
V = hsv[:,:,2]

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=0.8)
plt.subplot(421), plt.imshow(hsv), plt.title('HSV')
plt.subplot(423), plt.imshow(H), plt.title('H')
plt.subplot(425), plt.imshow(S), plt.title('S')
plt.subplot(427), plt.imshow(V), plt.title('V')

# RGB to HSV by hand
import numpy as np

def rgb2hsv(rgb):

    rgb = rgb.astype('float')
    maxv = np.amax(rgb, axis=2)/255
    maxc = np.argmax(rgb, axis=2)/255
    minv = np.amin(rgb, axis=2)/255
    minc = np.argmin(rgb, axis=2)/255 

    hsv = np.zeros(rgb.shape, dtype='float')
    hsv[maxc == minc, 0] = np.zeros(hsv[maxc == minc, 0].shape)
    hsv[maxc == 0, 0] = (((rgb[..., 1] - rgb[..., 2]) / (maxv - minv + np.spacing(1))) % 360 )[maxc == 0]
    hsv[maxc == 1, 0] = (((rgb[..., 2] - rgb[..., 0]) / (maxv - minv + np.spacing(1))) + 120.0)[maxc == 1]
    hsv[maxc == 2, 0] = (((rgb[..., 0] - rgb[..., 1]) / (maxv - minv + np.spacing(1))) + 240.0)[maxc == 2]
    hsv[maxv == 0, 1] = np.zeros(hsv[maxv == 0, 1].shape)
    hsv[maxv != 0, 1] = (1 - minv / (maxv + np.spacing(1)))[maxv != 0]
    hsv[..., 2] = maxv

    return hsv

hsv_by_hand = rgb2hsv(img)

H = hsv_by_hand[:,:,0]
S = hsv_by_hand[:,:,1]
V = hsv_by_hand[:,:,2]

plt.subplot(422), plt.imshow(hsv_by_hand), plt.title('hsv by hand')
plt.subplot(424), plt.imshow(H), plt.title('H by hand')
plt.subplot(426), plt.imshow(S), plt.title('S by hand')
plt.subplot(428), plt.imshow(V), plt.title('V by hand')
plt.show()
#-------------------------------------------------------------

# RGB to CMYK

# Convert RGB to CMYK by hand

img1 = img.astype(np.float64)/255.
K = 1 - np.max(img1, axis=2)
C = (1-img1[...,2] - K)/(1-K)
M = (1-img1[...,1] - K)/(1-K)
Y = (1-img1[...,0] - K)/(1-K)

cmyk = (np.dstack((C,M,Y,K)) * 255).astype(np.uint8)

plt.imshow(cmyk)
plt.title('cmyk by hand')
plt.show()

#---------------------------------------------------------

#Rgb to Lab using cv2
LAB_cv = cv.cvtColor(img, cv.COLOR_RGB2LAB)

#Convert RGB to lab by hand
def rgb2lab ( inputColor ) :

   num = 0
   RGB = [0, 0, 0]

   for value in inputColor :
       value = float(value) / 255

       if value > 0.04045 :
           value = ( ( value + 0.055 ) / 1.055 ) * 2.4
       else :
           value = value / 12.92

       RGB[num] = value * 100
       num = num + 1

   XYZ = [0, 0, 0,]

   X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
   Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
   Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
   XYZ[ 0 ] = round( X, 4 )
   XYZ[ 1 ] = round( Y, 4 )
   XYZ[ 2 ] = round( Z, 4 )

   XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
   XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
   XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

   num = 0
   for value in XYZ :

       if value > 0.008856 :
           value = value / ( 0.3333333333333333 )
       else :
           value = ( 7.787 * value ) + ( 16 / 116 )

       XYZ[num] = value
       num = num + 1

   Lab = [0, 0, 0]

   L = ( 116 * XYZ[ 1 ] ) - 16
   a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
   b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

   Lab [ 0 ] = round( L, 4 )
   Lab [ 1 ] = round( a, 4 )
   Lab [ 2 ] = round( b, 4 )

   return Lab

def COLOR_RGB2LAB(img):
    LAB = np.zeros([img.shape[0],img.shape[1],3])

    for hig in range(img.shape[0]):
        for wid in range(img.shape[1]):
            lab = rgb2lab(
                [img[hig,wid,2],
                img[hig,wid,1],
                img[hig,wid,0]]
            )
            # print(h,s,v)
            LAB[hig,wid,0] = lab[0]
            LAB[hig,wid,1] = lab[1]
            LAB[hig,wid,2] = lab[2]

    return LAB

lab_by_hand = COLOR_RGB2LAB(img)

plt.subplot(121), plt.imshow(LAB_cv), plt.title('LAB')
plt.subplot(122), plt.imshow(lab_by_hand), plt.title('LAB by hand')
plt.show()

cv.waitKey(0)



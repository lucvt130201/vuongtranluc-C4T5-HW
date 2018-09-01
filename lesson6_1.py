import cv2
import numpy as np
import matplotlib as plt

# cách gọi hàm từ  một file(3 cách)
# import test #import 1
# import test as t
# from test import *

# read image
I = cv2.imread("E:\\picture4.jpg")
Image_resize = cv2.resize(I, (600,600))
cv2.imshow("image", Image_resize)

# get row and col
row = I.shape[0]
cols = I.shape[1]
print("rows: ", row)
print("cols:", cols)
print("rgb: ", len(I.shape)) #kiểm tra ảnh có phải rgb ko?
#  nếu len() ==3 -> ảnh rgb, ngược lại là gray

# convert rgb to gray
gray = cv2.cvtColor(Image_resize, cv2.COLOR_RGB2GRAY)
cv2.imshow("gray", gray)
# convert gray to binary
thresh, binImg = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
cv2.imshow('binImg', binImg)

print("gray:", len(gray.shape))
print(len(binImg.shape))

cv2.waitKey()
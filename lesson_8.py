import cv2
import numpy as np

# read image

I = cv2.imread("image1.jpg")
cv2.imshow('image', I)
cv2.waitKey(1)

# convert image to gray
gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
gray_filter = gray.copy()

# Cách truy xuất dữ liệu ảnh
for row in range(10):
    for col in range(10):
        print(gray[row, col], end =' ')
    print()
#
# cách gán dữ liệu cho ảnh
for row in range(50,100):
    for col in range(20,50):
        gray[row, col] = 0
cv2.imshow('graynew', gray)
cv2.waitKey()
#
# implyment
def trung_binh_3x3(row,col,gray_filter):
    sumv = 0
    for i in range(row-1, row +2):
        for j in range(col-1, col+2):
            sumv = sumv + gray_filter[i,j]
    sumv = sumv/9.0
    return sumv

def median_filter(row, col, gray_filter):
    array_value =[]
    for i in range(row-1, row +2):
        for j in range(col-1, col+2):
            array_value.append(gray_filter[i,j])
    array_value = sorted(array_value)
    # for i in range(len(array_value)):
    #     print(array_value)
    return array_value[4]

rows = gray_filter.shape[0]
cols = gray_filter.shape[1]
gray_new = gray_filter.copy()
for row in range(1, rows -1):
    for cols in range(1,cols -1):
        gray_filter[row, cols] = trung_binh_3x3(row, cols, gray_filter)

cv2.imshow("gray after filter", gray_new)
cv2.waitKey()

rows = gray_filter.shape[0]
cols = gray_filter.shape[1]
gray_new2 = gray_filter.copy()
for row in range(1, rows -1):
    for cols in range(1 ,cols -1):
        gray_filter[row, cols] = median_filter(row,cols,gray_filter)

cv2.imshow("gray after filter 2", gray_new2)
cv2.waitKey()

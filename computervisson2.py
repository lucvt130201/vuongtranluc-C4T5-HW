import cv2
import numpy as np
import matplotlib as plt

# cách gọi hàm từ  một file(3 cách)
# import test #import 1
# import test as t
# from test import *

# read image
I = cv2.imread("E:\\picture4.jpg")
# cv2.imshow("image", I)
# resize Image
I_resize = cv2.resize(I, (300,300));
cv2.imshow("resize", I_resize)
cv2.waitKey()
# read image from webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow("video", frame)
    key = cv2.waitKey(30)
    if key&0xFF == ord ('q'):
        break



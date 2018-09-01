import cv2
import numpy as np
Image_one = cv2.imread('E:\ C4T\class\picture1.jpg')
covert = cv2.cvtColor(Image_one, cv2.COLOR_RGB2GRAY)
# # cv2.namedWindow(Image_one, cv2.WINDOW_NORMAL)
Image_one_resize = cv2.resize(Image_one, (600,800))
cv2.imshow("ảnh", Image_one_resize)
print(len(covert.shape))
print(covert.shape[1])
for i in range(covert.shape[0]):
    for j in range(covert.shape[1]):
        print(covert[i,j], end = " ")


cv2.waitKey()
cv2.destroyAllWindows()
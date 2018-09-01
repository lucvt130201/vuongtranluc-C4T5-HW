import cv2
import numpy as np

I = cv2.imread("image6.jpg")
cv2.imshow("image", I)
cv2.waitKey()

# extract r, g, b

B = I[:, :, 0] #B channel
G = I[:,:,1] # G channel
R = I[:,:,2] #R channel
cv2.imshow("Blue", B)
cv2.imshow("Green", G)
cv2.imshow("Red", R)
cv2.waitKey(1)

# combine red and blue
ImgBin = B&R
ImgRoi =~ImgBin # inverse Image
cv2.imshow("image 2",ImgRoi)
cv2.waitKey()

thresh,ImgRoi = cv2.threshold(ImgRoi, 50, 255, cv2.THRESH_BINARY)
# find contour
N, contour, hierachy = cv2.findContours(ImgRoi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for i in contour:
    cv2.drawContours(I, i, -1, (0,150,150), 3)

#     lấy chu vi
    chuvi = cv2.arcLength(i, True)

    #   lấy về diện tích
    dientich = cv2.contourArea(i, True)

#     xấp xỉ đường cong hay đa thức
    nedges = cv2.approxPolyDP(i, 5, True)
    print(len(nedges))
    (len(nedges))
    print(nedges)
    if (len(nedges) == 3):
        x = nedges [0] [0] [0]
        y = nedges [0] [0] [1]
        cv2.putText(I, "Triangle", (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,255,0))
    if (len(nedges) == 4):
        x = nedges [0] [0] [0]
        y = nedges [0] [0] [1]
        cv2.putText(I, "rectangle", (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,255,0))

cv2.imshow("contours", I)
cv2.waitKey()
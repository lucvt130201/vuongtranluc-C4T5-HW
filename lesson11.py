import cv2
import numpy as np

# # reread Image
# I = cv2.imread('E:\ C4T\class\DP7_EN_3.png')
# cv2.imshow("original", I)
# cv2.waitKey()

# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     cv2.imshow("video", frame)
#     key = cv2.waitKey(30)
#     if key&0xFF == ord('a'):
#         cv2.imwrite("image_capture1.jpg", frame)
#     if key&0xFF == ord('q'):
#         break

# # reread Image
I = cv2.imread('E:\ C4T\class\DP7_EN_3.png')
cv2.imshow("original", I)
cv2.waitKey()

# convert
gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
# cv2.imshow('gray image', gray2)
# cv2.waitKey()

# detect sift
sift = cv2.xfeatures2d.SIFT_create()
kp1, desl = sift.detectAndCompute(gray, None)
newImg = I.copy()
cv2.drawKeypoints(I, kp1, newImg)
cv2.imshow('new img', newImg)
cv2.waitKey()

#  load image 2
I2 = cv2.imread('E:\ C4T\class\mix.png')
cv2.imshow("original2", I2)
cv2.waitKey()

# convert
gray2 = cv2.cvtColor(I2, cv2.COLOR_RGB2GRAY)
# cv2.imshow('gray image2', gray2)
# cv2.waitKey()

# detect sift
sift2 = cv2.xfeatures2d.SIFT_create()
kp2, des2 = sift2.detectAndCompute(gray2, None)
newImg2 = I2.copy()
cv2.drawKeypoints(I2, kp2, newImg2)
cv2.imshow('new img2', newImg2)
cv2.waitKey()

# matchinh Image
bf = cv2.BFMatcher_create()  #bf means brute force
matches = bf.knnMatch(desl, des2, k = 2)
matchesImg = cv2.drawMatchesKnn(I, kp1, I2, kp2, matches, None)
cv2.imshow('matches', matchesImg)
cv2.waitKey()

# choose good point
good = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
goodMatch_img = cv2.drawMatches(I,kp1, I2, kp2, good, None)
cv2.imshow("goodmatches",goodMatch_img)
cv2.waitKey()

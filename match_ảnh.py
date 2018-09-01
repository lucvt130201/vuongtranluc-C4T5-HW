import cv2
import numpy as np

# load image insert
Img_insert = cv2.imread('E:\ C4T\class\DP7_EN_3.png')

# # reread Image
I = cv2.imread('E:\ C4T\class\dragonball.png')
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
mask = np.ones_like(I, dtype = np.float32)

#  load image 2
cap = cv2.VideoCapture(0)
while True:
    ret,I2 = cap.read()

    # convert
    gray2 = cv2.cvtColor(I2, cv2.COLOR_RGB2GRAY)
    # cv2.imshow('gray image2', gray2)
    # cv2.waitKey()

    # detect sift
    sift2 = cv2.xfeatures2d.SIFT_create()
    kp2, des2 = sift2.detectAndCompute(gray2, None)

    # matchinh Image
    bf = cv2.BFMatcher_create()  #bf means brute force
    matches = bf.knnMatch(desl, des2, k = 2)

    # choose good point
    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    # find homography
    src_points = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_points = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, H = cv2.findHomography(src_points, dst_points)
    w = gray.shape[1]
    h = gray.shape[0]
    pattern = cv2.resize(Img_insert, (w,h))

    ncorners = np.float32([[0,0], [w-1, 0], [w-1, h-1], [0, h-1]]).reshape(-1,1,2)
    if M is None:
        print('không tìm thấy M')
    else:
        npts = cv2.perspectiveTransform(ncorners, M)  # 4 điểm cũ nhân vs matran M tao điêm,r ms
        cv2.polylines(I2, np.int32([npts]), True,(0,0,255))

        # insert image
        blendmask = cv2.warpPerspective(mask, M, (I2.shape[1], I2.shape[0])) #warp ảnh insert vào trong I2
        newPattern = cv2.warpPerspective(pattern, M, (I2.shape[1], I2.shape[0]))
        img_result = I2 * (1 - blendmask) + newPattern
        img_result = cv2.convertScaleAbs(img_result)
        cv2.imshow('I2', I2)
        cv2.imshow('image results', img_result)


    goodMatch_img = cv2.drawMatches(I,kp1, I2, kp2, good, None)
    key = cv2.waitKey(1)
    if key&0xFF == ord('q'):
            break
    cv2.imshow("goodmatches",goodMatch_img)
    cv2.waitKey(1)
import cv2
import numpy
cap = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2 (1).xml")
while True:
    ret, frame = cap.read()

    # convert ảnh to hsv
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # thresh, bin = cv2.threshold(gray, 80, 255,cv2.THRESH_BINARY)
    # cv2.imshow("video", bin)
    faces = cascade.detectMultiScale(gray)
    for x, y, w, h  in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(30)
    if key&0xFF == ord ('q'):
        break
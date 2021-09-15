import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, frame = cap.read()
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_h, frame_s, frame_v = cv2.split(frame_hsv)
    v = frame_v.ravel()[np.flatnonzero(frame_v)]
    average_v = np.sum(v) / len(v)
    text = "Intensity: " + str(int(average_v))
    y0 = 50
    x0 = 50
    ret = cv2.putText(frame, text, (y0, x0),
                      cv2.FONT_HERSHEY_DUPLEX,
                      1, (10, 255, 10), 2)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = frame_gray.ravel()[np.flatnonzero(frame_gray)]
    average_gray = np.sum(gray) / len(gray)
    print(frame.shape, " Intensity:", average_v, " gray:", average_gray)
    cv2.imshow('title', ret)
    if cv2.waitKey(10) == 113:  # 点击q的时候退出
        break

import cv2
import numpy as np
import matplotlib.pyplot as plt
from urllib import request
import time

# im = cv2.imread("0.png")
# cv2.imshow("ss", im)
# cv2.waitKey()
cap0 = cv2.VideoCapture('D:/000_download/11/001.mp4')
cap1 = cv2.VideoCapture('D:/000_download/11/001.mp4')
cap2 = cv2.VideoCapture('D:/000_download/11/001.mp4')
cap3 = cv2.VideoCapture('D:/000_download/11/001.mp4')
cap4 = cv2.VideoCapture('D:/000_download/11/001.mp4')
cap5 = cv2.VideoCapture('D:/000_download/11/001.mp4')
cap6 = cv2.VideoCapture('D:/000_download/11/001.mp4')
cap7 = cv2.VideoCapture('D:/000_download/11/001.mp4')
_, frame = cap0.read()

while cap0.isOpened():
    _, frame = cap0.read()
    # plt.imshow(frame)
    # plt.show()

    frame_resize = cv2.resize(frame, (800, 400), cv2.INTER_CUBIC)
    # plt.imshow(frame_resize)
    # plt.show()
    # time.sleep(10)
    # cv2.imshow("nihao", frame_resize)

    # if ii // 8
    frame_hsv = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2HSV)
    frame_h, frame_s, frame_v = cv2.split(frame_hsv)
    v = frame_v.ravel()[np.flatnonzero(frame_v)]
    average_v = np.sum(v) / len(v)
    text = "Intensity: " + str(int(average_v))
    y0 = 50
    x0 = 50
    # frame_resize = cv2.resize(frame, (800, 400), cv2.INTER_CUBIC)
    ret = cv2.putText(frame_resize, text, (y0, x0),
                      cv2.FONT_HERSHEY_DUPLEX,
                      1, (10, 255, 10), 2)
    #
    # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = frame_gray.ravel()[np.flatnonzero(frame_gray)]
    # average_gray = np.sum(gray) / len(gray)
    # print(frame.shape, " Intensity:", average_v, " gray:", average_gray)

    cv2.imshow('title', ret)

    if cv2.waitKey(10) == 113:  # 点击q的时候退出
        break



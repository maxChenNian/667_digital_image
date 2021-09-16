"""
工程项目：
1. 监控摄像单帧图像亮度计算

功能模块：
1. logitech 摄像头的视频读取和显示
2. opencv 设置显示窗口的大小和位置
"""

import cv2
import numpy as np

# cap0 = cv2.VideoCapture('D:/000_download/11/001.mp4')

cap1 = cv2.VideoCapture(0)
# cap2 = cv2.VideoCapture(0)

if cap1.isOpened():
    width = cap1.get(3)
    height = cap1.get(4)
    rate = cap1.get(5)
    print(width, height, rate)
y0 = 50
x0 = 50

for i in range(11, -1, -1):
    window_name = str(i + 1)
    cv2.namedWindow(window_name, 0)
    cv2.resizeWindow(window_name, 480, 340)
    col = i % 4 * 480
    row = i // 4 * 340
    cv2.moveWindow(window_name, col, row)

iii = 0
average_v = 0
while cap1.isOpened():
    if iii == 12:
        iii = 0

    _, frame = cap1.read()
    if iii == 0:
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, _, frame_v = cv2.split(frame_hsv)
        average_v = np.mean(frame_v.astype("float32")) // 32
    else:
        average_v = average_v

    text = "Intensity: " + str(int(average_v)) + " level"
    ret = cv2.putText(frame, text, (y0, x0),
                      cv2.FONT_HERSHEY_DUPLEX,
                      1, (10, 255, 10), 1)

    for i in range(12):
        cv2.imshow(str(i + 1), ret)

    if cv2.waitKey(10) == 113:  # 点击q的时候退出
        cv2.destroyAllWindows()
        break

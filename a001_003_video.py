"""
工程项目：
1. 监控摄像单帧图像亮度计算

功能模块：
1. threading 多线程进行图像处理，提升视频显示的速度
2. opencv 将图像帧存储成视频格式
3. exec 执行字符串代码行，目的按顺序生成新变量名
"""

import cv2
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import time
import threading


def brightness(frame):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, _, frame_v = cv2.split(frame_hsv)
    # average_v = np.mean(frame_v.astype("float32")) // 32
    average_v = np.mean(frame_v.astype("float32"))
    return average_v


def video_imshow(video_path, window_num):
    cap = cv2.VideoCapture(str(video_path))
    # y0 = 10
    # x0 = 80

    window_name = str(window_num + 1)
    cv2.namedWindow(window_name, 0)
    cv2.resizeWindow(window_name, 480, 340)
    col = window_num % 4 * 480
    row = window_num // 4 * 340
    cv2.moveWindow(window_name, col, row)

    while cap.isOpened():

        _, frame_ori = cap.read()
        frame = cv2.resize(frame_ori, (480, 340), cv2.INTER_CUBIC)

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, _, frame_v = cv2.split(frame_hsv)
        heat_img = cv2.applyColorMap(frame_v, cv2.COLORMAP_JET)

        aaa = frame_v + 150
        heat_img2 = cv2.applyColorMap(frame_v + 150, cv2.COLORMAP_JET)

        for ii in range(16):
            row_a = ii // 4
            col_b = ii % 4

            average_v = brightness(frame[row_a * 85:(row_a + 1) * 85,
                                   col_b * 120:(col_b + 1) * 120, :])
            text = str(int(average_v))
            # text = "Intensity: " + str(int(average_v)) + " level"

            # cv2.rectangle(heat_img, (col_b * 120, row_a * 85),
            #               ((col_b + 1) * 120, (row_a + 1) * 85), (0, 255, 0), 1)
            # cv2.putText(heat_img, text, (col_b * 120, row_a * 85 + 50),
            #             cv2.FONT_HERSHEY_DUPLEX,
            #             1, (10, 255, 10), 1)

            cv2.rectangle(heat_img, (col_b * 120, row_a * 85),
                          ((col_b + 1) * 120, (row_a + 1) * 85), (0, 0, 0), 1)
            cv2.putText(heat_img, text, (col_b * 120, row_a * 85 + 50),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1, (0, 0, 0), 1)

        # cv2.imshow(str(window_num + 1), heat_img)
        if 1 <= window_num + 1 <= 4:
            cv2.imshow(str(window_num + 1), heat_img)
        elif window_num + 1 == 6:
            cv2.imshow(str(window_num + 1), heat_img2)

        else:
            cv2.imshow(str(window_num + 1), frame)

        if cv2.waitKey(10) == 113:  # 点击q的时候退出
            # cv2.destroyAllWindows()
            cv2.destroyWindow(winname=str(window_num + 1))
            break


def output_video(video_path, window_num):
    cap = cv2.VideoCapture(str(video_path))
    ww = int(cap.get(3))
    hh = int(cap.get(4))
    fps = int(cap.get(5))
    print(ww, hh, fps)
    name = str(window_num + 1) + ".avi"
    # fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(name, fourcc, fps, (ww, hh))
    y0 = 10
    x0 = 50
    while cap.isOpened():
        _, frame = cap.read()
        # print(frame_ori.shape)
        # frame = cv2.resize(frame_ori, (480, 340), cv2.INTER_CUBIC)
        average_v = brightness(frame)
        text = "Intensity: " + str(int(average_v))
        cv2.rectangle(frame, (0, 0), (300, 300), (0, 255, 0), 4)
        # cv2.putText(frame, text, (y0, x0),
        #             cv2.FONT_HERSHEY_DUPLEX,
        #             1, (10, 255, 10), 2)
        # nihao = np.zeros((200, 200)).astype("uint8")
        video.write(frame)
        # iiii += 1
        # if iiii == 200:
        #     break


class myThread(threading.Thread):
    def __init__(self, threadID, video_path, window_num):
        threading.Thread.__init__(self)
        self.threadID = str(threadID)
        self.video_path = video_path
        self.window_num = window_num

    def run(self):
        print("开始线程：" + self.threadID)
        video_imshow(self.video_path, self.window_num)
        # output_video(self.video_path, self.window_num)
        print("退出线程：" + self.threadID)


# if __name__ == "__main__":
#
#     # 获取视频
#     mp4_path = 'D:/000_download/11'
#     mp4_file_list = sorted(pathlib.Path(mp4_path).glob("*.mp4"))
#     cap_list = []
#
#     for i, path in enumerate(mp4_file_list):
#         # print(f'{"thread" + str(i + 1)} = myThread(1, str(path), {str(i)})')
#         exec(f'{"thread" + str(i + 1)} = myThread(1, str(path), {str(i)})')
#         print(f'{"thread" + str(i + 1)}.start()')
#
#     for i in range(len(mp4_file_list)):
#         exec(f'{"thread" + str(len(mp4_file_list) - i)}.start()')
#
#     for i in range(len(mp4_file_list)):
#         exec(f'{"thread" + str(i + 1)}.join()')

if __name__ == "__main__":
    # 测试图像
    pass


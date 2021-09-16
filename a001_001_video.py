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


# 建立opencv窗口
for i in range(11, -1, -1):
    window_name = str(i + 1)
    cv2.namedWindow(window_name, 0)
    cv2.resizeWindow(window_name, 480, 340)
    col = i % 4 * 480
    row = i // 4 * 340
    cv2.moveWindow(window_name, col, row)

# 获取视频
mp4_path = 'D:/000_download/11'
mp4_file_list = sorted(pathlib.Path(mp4_path).glob("*.mp4"))
cap_list = []
for i, path in enumerate(mp4_file_list):
    print(str(path))
    # cap_var_name = 'cap' + str(i)
    # exec(f'{cap_var_name} = cv2.VideoCapture(str(path))')
    cap_list.append(cv2.VideoCapture(str(path)))

y0 = 10
x0 = 80
iii = 0
average_v_list = [0, 0, 0, 0, 0, 0, 0, 0]
while cap_list[0].isOpened():

    ret_list = []
    start_time = time.time()
    for i, cap in enumerate(cap_list):
        _, frame_ori = cap.read()
        frame = cv2.resize(frame_ori, (480, 340), cv2.INTER_CUBIC)
        if iii % 8 == i:
            average_v_list[i] = brightness(frame)
            average_v = average_v_list[i]
        else:
            average_v = average_v_list[i]

        text = "Intensity: " + str(int(average_v)) + " level"
        ret = cv2.putText(frame, text, (y0, x0),
                          cv2.FONT_HERSHEY_DUPLEX,
                          1, (10, 255, 10), 1)
        cv2.imshow(str(i + 1), ret)
    print(time.time() - start_time)

    iii += 1

    if cv2.waitKey(10) == 113:  # 点击q的时候退出
        cv2.destroyAllWindows()
        break

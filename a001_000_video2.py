"""
工程项目：
1. 监控摄像单帧图像亮度计算

功能模块：
1. logitech 摄像头的视频读取和显示
2. opencv 设置显示窗口的大小和位置
3. 将此代码写成多线程显示的形式
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


def video_imshow(window_num, cap):
    # cap = cv2.VideoCapture(str(video_path))
    cap = cap
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
        if 1 <= window_num + 1 <= 2:
            cv2.imshow(str(window_num + 1), heat_img)

        elif window_num + 1 == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            heat_img2 = cv2.applyColorMap(frame_gray, cv2.COLORMAP_JET)
            cv2.imshow(str(window_num + 1), heat_img2)
        else:
            average_v = brightness(frame)
            text = str(int(average_v))
            cv2.putText(frame, text, (200, 200),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1, (10, 255, 10), 1)

            cv2.imshow(str(window_num + 1), frame)

        if cv2.waitKey(10) == 113:  # 点击q的时候退出
            # cv2.destroyAllWindows()
            cv2.destroyWindow(winname=str(window_num + 1))
            break


class myThread(threading.Thread):
    def __init__(self, threadID, cap, window_num):
        threading.Thread.__init__(self)
        self.threadID = str(threadID)
        self.cap = cap
        self.window_num = window_num

    def run(self):
        print("开始线程：" + self.threadID)
        video_imshow(self.window_num, self.cap)
        # output_video(self.video_path, self.window_num)
        print("退出线程：" + self.threadID)


if __name__ == "__main__":
    VideoCaptureProperties = []
    # print(len(VideoCaptureProperties))
    VideoCAPtureProperties_dict = {
        "CAP_PROP_POS_MSEC       ": 0,
        # !< Current position of the video file in milliseconds.
        "CAP_PROP_POS_FRAMES     ": 1,
        # !< 0-based index of the frame to be decoded/"CAPtured next.
        "CAP_PROP_POS_AVI_RATIO  ": 2,
        # !< Relative position of the video file: 0":start of the film, 1":end of the film.
        "CAP_PROP_FRAME_WIDTH    ": 3,
        # !< Width of the frames in the video stream.
        "CAP_PROP_FRAME_HEIGHT   ": 4,
        # !< Height of the frames in the video stream.
        "CAP_PROP_FPS            ": 5,  # !< Frame rate.
        "CAP_PROP_FOURCC         ": 6,
        # !< 4-character code of codec. see VideoWriter::fourcc .
        "CAP_PROP_FRAME_COUNT    ": 7,  # !< Number of frames in the video file.
        "CAP_PROP_FORMAT         ": 8,
        # !< Format of the %Mat objects (see Mat::type()) returned by Video"CAPture::retrieve().
        # !< Set value -1 to fetch undecoded RAW video streams (as Mat 8UC1).
        "CAP_PROP_MODE           ": 9,
        # !< Backend-specific value indicating the current "CAPture mode.
        "CAP_PROP_BRIGHTNESS    ": 10,
        # !< Brightness of the image (only for those cameras that support).
        "CAP_PROP_CONTRAST      ": 11,
        # !< Contrast of the image (only for cameras).
        "CAP_PROP_SATURATION    ": 12,
        # !< Saturation of the image (only for cameras).
        "CAP_PROP_HUE           ": 13,
        # !< Hue of the image (only for cameras).
        "CAP_PROP_GAIN          ": 14,
        # !< Gain of the image (only for those cameras that support).
        "CAP_PROP_EXPOSURE      ": 15,
        # !< Exposure (only for those cameras that support).
        "CAP_PROP_CONVERT_RGB   ": 16,
        # !< Boolean flags indicating whether images should be converted to RGB. <br/>
        # !< *GStreamer note*: The flag is ignored in case if custom pipeline is used. It's user responsibility to interpret pipeline output.
        "CAP_PROP_WHITE_BALANCE_BLUE_U ": 17,  # !< Currently unsupported.
        "CAP_PROP_RECTIFICATION ": 18,
        # !< Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently).
        "CAP_PROP_MONOCHROME    ": 19,
        "CAP_PROP_SHARPNESS     ": 20,
        "CAP_PROP_AUTO_EXPOSURE ": 21,
        # !< DC1394: exposure control done by camera, user can adjust reference level using this feature.
        "CAP_PROP_GAMMA         ": 22,
        "CAP_PROP_TEMPERATURE   ": 23,
        "CAP_PROP_TRIGGER       ": 24,
        "CAP_PROP_TRIGGER_DELAY ": 25,
        "CAP_PROP_WHITE_BALANCE_RED_V ": 26,
        "CAP_PROP_ZOOM          ": 27,
        "CAP_PROP_FOCUS         ": 28,
        "CAP_PROP_GUID          ": 29,
        "CAP_PROP_ISO_SPEED     ": 30,
        "CAP_PROP_BACKLIGHT     ": 32,
        "CAP_PROP_PAN           ": 33,
        "CAP_PROP_TILT          ": 34,
        "CAP_PROP_ROLL          ": 35,
        "CAP_PROP_IRIS          ": 36,
        "CAP_PROP_SETTINGS      ": 37,
        # !< Pop up video/camera filter dialog (note: only supported by DSHOW backend currently. The property value is ignored)
        "CAP_PROP_BUFFERSIZE    ": 38,
        "CAP_PROP_AUTOFOCUS     ": 39,
        "CAP_PROP_SAR_NUM       ": 40,  # !< Sample aspect ratio: num/den (num)
        "CAP_PROP_SAR_DEN       ": 41,  # !< Sample aspect ratio: num/den (den)
        "CAP_PROP_BACKEND       ": 42,
        # !< Current backend (enum Video"CAPtureAPIs). Read-only property
        "CAP_PROP_CHANNEL       ": 43,
        # !< Video input or Channel Number (only for those cameras that support)
        "CAP_PROP_AUTO_WB       ": 44,  # !< enable/ disable auto white-balance
        "CAP_PROP_WB_TEMPERATURE": 45,  # !< white-balance color temperature
        "CAP_PROP_CODEC_PIXEL_FORMAT ": 46,
        # !< (read-only) codec's pixel format. 4-character code - see VideoWriter::fourcc . Subset of [AV_PIX_FMT_*](https:#github.com/FFmpeg/FFmpeg/blob/master/libavcodec/raw.c) or -1 if unknown
        "CAP_PROP_BITRATE       ": 47,
        # !< (read-only) Video bitrate in kbits/s
        "CAP_PROP_ORIENTATION_META": 48,
        # !< (read-only) Frame rotation defined by stream meta (applicable for FFmpeg back-end only)
        "CAP_PROP_ORIENTATION_AUTO": 49,
        # !< if true - rotates output frames of Cv"CAPture considering video file's metadata  (applicable for FFmpeg back-end only) (https:#github.com/opencv/opencv/issues/15499)
        "CAP_PROP_HW_ACCELERATION": 50,
        # !< (**open-only**) Hardware acceleration type (see #VideoAccelerationType). Setting supported only via `params` parameter in cv::Video"CAPture constructor / .open() method. Default value is backend-specific.
        "CAP_PROP_HW_DEVICE      ": 51,
        # !< (**open-only**) Hardware device index (select GPU if multiple available). Device enumeration is acceleration type specific.
        "CAP_PROP_HW_ACCELERATION_USE_OPENCL": 52,
        # !< (**open-only**) If non-zero, create new OpenCL context and bind it to current thread. The OpenCL context created with Video Acceleration context attached it (if not attached yet) for optimized GPU data copy between HW accelerated decoder and cv::UMat.
        "CAP_PROP_OPEN_TIMEOUT_MSEC": 53,
        # !< (**open-only**) timeout in milliseconds for opening a video "CAPture (applicable for FFmpeg back-end only)
        "CAP_PROP_READ_TIMEOUT_MSEC": 54,
        # !< (**open-only**) timeout in milliseconds for reading from a video "CAPture (applicable for FFmpeg back-end only)
        "CAP_PROP_STREAM_OPEN_TIME_USEC ": 55,
        # <! (read-only) time in microseconds since Jan 1 1970 when stream was opened. Applicable for FFmpeg backend only. Useful for RTSP and other live streams
    }
    for item in VideoCAPtureProperties_dict:
        # print(item)
        VideoCaptureProperties.append(item)

    # 获取视频
    # mp4_path = 'D:/000_download/11'
    # mp4_file_list = sorted(pathlib.Path(mp4_path).glob("*.mp4"))
    # cap_list = []

    cap = cv2.VideoCapture(0)
    cap.set(15, -12)
    # cap.set(23, 1000)
    for i in range(47):
        print(f"No.={i} parameter:{VideoCaptureProperties[i]} {cap.get(i)}")

    for i in range(4):
        # print(f'{"thread" + str(i + 1)} = myThread(1, str(path), {str(i)})')
        exec(f'{"thread" + str(i + 1)} = myThread(1, cap, {str(i)})')
        print(f'{"thread" + str(i + 1)}.start()')

    for i in range(4):
        exec(f'{"thread" + str(4 - i)}.start()')

    for i in range(4):
        exec(f'{"thread" + str(i + 1)}.join()')

# cap1 = cv2.VideoCapture(0)

# if cap1.isOpened():
#     width = cap1.get(3)
#     height = cap1.get(4)
#     rate = cap1.get(5)
#     print(width, height, rate)
# y0 = 50
# x0 = 50
#
# for i in range(11, -1, -1):
#     window_name = str(i + 1)
#     cv2.namedWindow(window_name, 0)
#     cv2.resizeWindow(window_name, 480, 340)
#     col = i % 4 * 480
#     row = i // 4 * 340
#     cv2.moveWindow(window_name, col, row)
#
# iii = 0
# average_v = 0
# while cap1.isOpened():
#     if iii == 12:
#         iii = 0
#
#     _, frame = cap1.read()
#     if iii == 0:
#         frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         _, _, frame_v = cv2.split(frame_hsv)
#         average_v = np.mean(frame_v.astype("float32")) // 32
#     else:
#         average_v = average_v
#
#     text = "Intensity: " + str(int(average_v)) + " level"
#     ret = cv2.putText(frame, text, (y0, x0),
#                       cv2.FONT_HERSHEY_DUPLEX,
#                       1, (10, 255, 10), 1)
#
#     for i in range(12):
#         cv2.imshow(str(i + 1), ret)
#
#     if cv2.waitKey(10) == 113:  # 点击q的时候退出
#         cv2.destroyAllWindows()
#         break

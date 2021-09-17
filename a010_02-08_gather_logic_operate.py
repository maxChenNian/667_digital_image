"""
功能模块：
1. opencv 图像集合和逻辑操作
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

body_scan_im = cv2.imread(
    "a000_000_Digital_image_processing_image/DIP3E_Original_Images_CH02/Fig0232(a)(partial_body_scan).tif")

body_scan_im_contrary = 255 - body_scan_im




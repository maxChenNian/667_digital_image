"""
功能模块：
1. opencv 添加高斯噪声，模拟低照度情况下图像的质量
1. 对噪声图像进行去噪，相加再平均
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

im_ori_BRG = cv2.imread(
    "a000_000_Digital_image_processing_image/DIP3E_Original_Images_CH02/Fig0226(galaxy_pair_original).tif")
im_ori_gray = cv2.cvtColor(im_ori_BRG, cv2.COLOR_BGR2GRAY)

rows, cols = np.shape(im_ori_gray)
gauss_array = np.random.rand(rows, cols)


for i in range(20):
    print(random.gauss(0, 64))

aa = np.random.rand(128, 128) * 255
plt.imshow(aa.astype("uint8"))
plt.show()

plt.imshow(im_ori_gray)
plt.axis("off")
plt.show()

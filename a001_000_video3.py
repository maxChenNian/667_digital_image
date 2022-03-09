"""
工程项目：
1. 监控摄像单帧图像亮度计算

功能模块：
1. 按照公式计算相机曝光值
"""
import numpy as np

# 焦距 mm
f = 4.73

# 光圈大小，手机镜头和摄像头镜头的光圈一般都是固定的
F = f / 1.8

# 快门速度
speed = 750

# 曝光时间
S = 1 / 750

# 相机的感光度
ISO = 100

# ISO不变的情况下计算相机的感光度
for s in range(750, 0, -50):
    S = 1 / s
    EV = 2 * np.log2(F) - np.log2(S) + np.log2(ISO / 100)
    print(EV)

for s in range(100, 0, -10):
    S = 1 / (s / 100)
    EV = 2 * np.log2(F) - np.log2(S) + np.log2(ISO / 100)
    print(EV)

EV = 2 * np.log2(F) - np.log2(1 / 1) + np.log2(ISO / 100)
print(EV)

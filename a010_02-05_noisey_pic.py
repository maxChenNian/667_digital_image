"""
功能模块：
1. opencv 添加高斯噪声，模拟低照度情况下图像的质量
1. 对噪声图像进行去噪，相加再平均
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


# 默认 均值Mu=0
def add_gauss_noise(im_gray, sigma=8, k=1):
    # k 添加噪声的次数
    rows, cols = np.shape(im_ori_gray)
    gauss_array = (np.random.rand(rows, cols) * 2 - 1) * sigma
    nosie_im = im_gray.astype("float32")

    for i in range(k):
        nosie_im = nosie_im + gauss_array

    return nosie_im.astype("uint8")


# 相加再平均对图像进行去噪
def add_mean_noise(im_gray, sigma=8, k=1, k2=1):
    im_noise = add_gauss_noise(im_gray, sigma, k)
    # k 相加再平均的次数
    add_im_noise = im_noise.astype("float32")
    for i in range(k2):
        add_im_noise = add_im_noise + \
                       add_gauss_noise(im_gray, sigma, k).astype("float32")

    mean_im_noise = add_im_noise / k
    return mean_im_noise.astype("uint8")


im_ori_BRG = cv2.imread(
    "a000_000_Digital_image_processing_image/DIP3E_Original_Images_CH02/Fig0226(galaxy_pair_original).tif")
im_ori_gray = cv2.cvtColor(im_ori_BRG, cv2.COLOR_BGR2GRAY)

mu = 0
sigma = 20

nums = []
for i in range(200):
    nums.append(random.gauss(mu, sigma))

galaxy_noise_im = add_gauss_noise(im_ori_gray, 64, 1)
galaxy_denoise_im = add_mean_noise(im_ori_gray, 64, 1, k2=100)

# 确定显示的画幅
fig, axs = plt.subplots(nrows=2,
                        ncols=2,
                        tight_layout=True,
                        num="IMAGE_PREPROCESS_METHOD",  # 整幅图的名字，可以使用数字
                        figsize=(7, 5),  # 整幅图物理尺寸，宽高单位英寸
                        dpi=200  # 每英寸物理尺寸含有的像素点的数量
                        )
axs[0, 0].plot(nums)
axs[0, 1].hist(nums)
axs[1, 0].imshow(cv2.cvtColor(galaxy_noise_im, cv2.COLOR_GRAY2RGB))
axs[1, 0].set_title("mu=0 sigma=64")
axs[1, 0].axis("off")
axs[1, 1].imshow(cv2.cvtColor(galaxy_denoise_im, cv2.COLOR_GRAY2RGB))
axs[1, 1].set_title("mu=0 sigma=64")
axs[1, 1].axis("off")
plt.show()

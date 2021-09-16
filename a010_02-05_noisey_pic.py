"""
功能模块：
1. opencv 添加高斯噪声，模拟低照度情况下图像的质量
1. 对噪声图像进行去噪，相加再平均
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.gridspec as gridspec


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

    mean_im_noise = add_im_noise / k2
    return mean_im_noise.astype("uint8")


im_ori_BRG = cv2.imread(
    "a000_000_Digital_image_processing_image/DIP3E_Original_Images_CH02/Fig0226(galaxy_pair_original).tif")
im_ori_gray = cv2.cvtColor(im_ori_BRG, cv2.COLOR_BGR2GRAY)
galaxy_noise_im = add_gauss_noise(im_ori_gray, 64, 1)

mu = 0
sigma = 20

nums = []
for i in range(200):
    nums.append(random.gauss(mu, sigma))

# 确定显示的画幅
fig0 = plt.figure(num="IMAGE_PREPROCESS_METHOD",  # 整幅图的名字，可以使用数字
                  figsize=(8, 4),  # 整幅图物理尺寸，宽高单位英寸
                  dpi=200)
gs = gridspec.GridSpec(2, 4, figure=fig0)

ax1 = fig0.add_subplot(gs[0, 0:2])
ax1.plot(nums)
ax1.set_title("random gauss nums", fontsize='medium')

ax2 = fig0.add_subplot(gs[1, 0:2])
ax2.hist(nums, bins=50)
ax2.set_title("random gauss distributed", fontsize='medium')

ax3 = fig0.add_subplot(gs[0:2, 2:4])
ax3.set_title("galaxy original pic", fontsize='medium')
ax3.imshow(im_ori_BRG)
ax3.axis("off")
# plt.savefig("a000_001_output_image/a010_02-05_galaxy_gauss.tif")
plt.show()

# 确定显示的画幅
fig, axs = plt.subplots(nrows=2,
                        ncols=3,
                        tight_layout=True,
                        num="IMAGE_PREPROCESS_METHOD",  # 整幅图的名字，可以使用数字
                        figsize=(5, 4),  # 整幅图物理尺寸，宽高单位英寸
                        dpi=200  # 每英寸物理尺寸含有的像素点的数量
                        )
k2_list = [5, 10, 20, 50, 100]
for i in range(0, 6, 1):
    if i == 0:
        axs[i // 3, i % 3].imshow(
            cv2.cvtColor(galaxy_noise_im, cv2.COLOR_GRAY2RGB))
        axs[i // 3, i % 3].set_title("noise im", fontsize='medium')
        # axs[i // 3, i % 3].text(x=10, y=40, s="mu=0 sigma=64", fontsize='medium',
        #                         color="black",
        #                         bbox=dict(facecolor='1', edgecolor='none', pad=1))
        axs[i // 3, i % 3].axis("off")
    else:
        axs[i // 3, i % 3].imshow(
            cv2.cvtColor(add_mean_noise(im_ori_gray, 64, 1, k2=k2_list[i - 1]),
                         cv2.COLOR_GRAY2RGB))
        axs[i // 3, i % 3].set_title(f"mean_nums={str(k2_list[i - 4])}",
                                     fontsize='medium')
        axs[i // 3, i % 3].axis("off")
plt.savefig("a000_001_output_image/a010_02-05_noisey_pic.tif")
plt.show()

"""
功能模块：
1. opencv 使用图像相乘来矫正图像的阴影
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

im_ori = cv2.imread(
    "a000_000_Digital_image_processing_image/DIP3E_Original_Images_CH02/Fig0229(a)(tungsten_filament_shaded).tif")

im_mask = cv2.imread(
    "a000_000_Digital_image_processing_image/DIP3E_Original_Images_CH02/Fig0229(b)(tungsten_sensor_shading).tif")

im_ori_float = im_ori[:, :, 0].astype("float")
im_mask_float = im_mask[:, :, 0].astype("float") / 255
im_new = im_ori_float * im_mask_float
im_new = im_new / np.max(im_new) * 255

# 直方图均衡化,图像增强
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_equalized = clahe.apply(im_new.astype("uint8"))

# 确定显示的画幅
fig, axs = plt.subplots(nrows=1,
                        ncols=4,
                        tight_layout=True,
                        num="IMAGE_PREPROCESS_METHOD",  # 整幅图的名字，可以使用数字
                        figsize=(8, 2.5),  # 整幅图物理尺寸，宽高单位英寸
                        dpi=200  # 每英寸物理尺寸含有的像素点的数量
                        )
axs[0].imshow(im_ori)
axs[0].set_title("original pic", fontsize='medium')
axs[0].axis("off")
axs[1].imshow(im_mask)
axs[1].set_title("mask pic", fontsize='medium')
axs[1].axis("off")
axs[2].imshow(cv2.cvtColor(im_new.astype("uint8"), cv2.COLOR_GRAY2RGB))
axs[2].set_title("output pic", fontsize='medium')
axs[2].axis("off")
axs[3].imshow(cv2.cvtColor(img_equalized, cv2.COLOR_GRAY2RGB))
axs[3].set_title("enhancement pic", fontsize='medium')
axs[3].axis("off")
plt.savefig("a000_001_output_image/a010_02-07_image_multiply1.tif")
plt.show()

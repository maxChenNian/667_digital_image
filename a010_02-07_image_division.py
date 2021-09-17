"""
功能模块：
1. opencv 利用mask提取图像中所需要的区域
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

dental_xray_im = cv2.imread(
    "a000_000_Digital_image_processing_image/DIP3E_Original_Images_CH02/Fig0230(a)(dental_xray).tif")
dental_xray_mask = cv2.imread(
    "a000_000_Digital_image_processing_image/DIP3E_Original_Images_CH02/Fig0230(b)(dental_xray_mask).tif")

dental_xray_im_float = dental_xray_im[:, :, 0].astype("float")
dental_xray_mask_float = dental_xray_mask[:, :, 0].astype("float") / 255
mul_im = dental_xray_im_float * dental_xray_mask_float
mul_im = mul_im - np.min(mul_im)
mul_im = mul_im / np.max(mul_im) * 255


# 确定显示的画幅
fig, axs = plt.subplots(nrows=1,
                        ncols=3,
                        tight_layout=True,
                        num="IMAGE_PREPROCESS_METHOD",  # 整幅图的名字，可以使用数字
                        figsize=(6, 1.8),  # 整幅图物理尺寸，宽高单位英寸
                        dpi=200  # 每英寸物理尺寸含有的像素点的数量
                        )
axs[0].imshow(dental_xray_im)
axs[0].set_title("original pic", fontsize='medium')
axs[0].axis("off")
axs[1].imshow(dental_xray_mask)
axs[1].set_title("mask pic", fontsize='medium')
axs[1].axis("off")
axs[2].imshow(cv2.cvtColor(mul_im.astype("uint8"), cv2.COLOR_GRAY2RGB))
axs[2].set_title("output pic", fontsize='medium')
axs[2].axis("off")
plt.savefig("a000_001_output_image/a010_02-07_image_multiply2.tif")
plt.show()
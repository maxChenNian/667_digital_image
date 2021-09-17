"""
功能模块：
1. opencv 对提取血管造影
2. 自适应直方图均衡化
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

angiography_mask_im = cv2.imread(
    "a000_000_Digital_image_processing_image/DIP3E_Original_Images_CH02/Fig0228(a)(angiography_mask_image).tif")

angiography_live_im = cv2.imread(
    "a000_000_Digital_image_processing_image/DIP3E_Original_Images_CH02/Fig0228(b)(angiography_live_ image).tif")

angiography_im = angiography_live_im.astype(
    'float32') - angiography_mask_im.astype('float32')
angiography_im = angiography_im + np.abs(np.min(np.min(angiography_im)))
angiography_im = angiography_im / (np.max(np.max(angiography_im))) * 255

# plt.imshow(angiography_im.astype("uint8"))
# plt.show()

# 直方图均衡化,图像增强
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_equalized = clahe.apply(angiography_im[:, :, 0].astype("uint8"))

# plt.imshow(cv2.cvtColor(img_equalized, cv2.COLOR_GRAY2RGB))
# plt.show()

# 确定显示的画幅
fig, axs = plt.subplots(nrows=1,
                        ncols=4,
                        tight_layout=True,
                        num="IMAGE_PREPROCESS_METHOD",  # 整幅图的名字，可以使用数字
                        figsize=(6, 1.8),  # 整幅图物理尺寸，宽高单位英寸
                        dpi=200  # 每英寸物理尺寸含有的像素点的数量
                        )
axs[0].imshow(angiography_mask_im)
axs[0].set_title("mask image", fontsize='medium')
axs[0].axis("off")
axs[1].imshow(angiography_live_im)
axs[1].set_title("live image", fontsize='medium')
axs[1].axis("off")
axs[2].imshow(angiography_im.astype("uint8"))
axs[2].set_title("angiography", fontsize='medium')
axs[2].axis("off")
axs[3].imshow(cv2.cvtColor(img_equalized, cv2.COLOR_GRAY2RGB))
axs[3].set_title("angio enhancement", fontsize='medium')
axs[3].axis("off")
plt.savefig("a000_001_output_image/a010_02-06_image_enhancement2.tif")
plt.show()

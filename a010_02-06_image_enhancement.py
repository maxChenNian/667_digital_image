"""
功能模块：
1. 对 8bits 灰度图进行分割，分成 1bit-8bit 二值图
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

im = cv2.imread(
    "a000_000_Digital_image_processing_image/DIP3E_Original_Images_CH02/Fig0227(a)(washington_infrared).tif")

im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
rows, cols = np.shape(im_gray)
im_gray_float = im_gray.astype('float32')

im_bit_list = []
im_gray_left_list = []
im_add_mask_list = []

im_add_mask = np.zeros((rows, cols))
im_add_mask_list.append(im_add_mask)

for i in range(8, 0, -1):
    if i == 1:
        im_bit = im_gray_float % 2
        im_gray_left = im_gray_float - im_add_mask
    else:
        im_bit = (im_gray_float - im_add_mask) // (2 ** (i - 1))
        im_gray_left = im_gray_float - im_add_mask
        im_add_mask = im_add_mask + im_bit * (2 ** (i - 1))

    im_add_mask_list.append(im_add_mask)
    im_bit_list.append(im_bit)
    im_gray_left_list.append(im_gray_left)
    # plt.imshow(cv2.cvtColor((im_bit * 255).astype("uint8"), cv2.COLOR_GRAY2RGB))
    # plt.imshow(cv2.cvtColor(im_gray_left.astype("uint8"), cv2.COLOR_GRAY2RGB))
    # plt.imshow(cv2.cvtColor(im_add_mask.astype("uint8"), cv2.COLOR_GRAY2RGB))
    # plt.show()

# 确定显示的画幅
fig, axs = plt.subplots(nrows=2,
                        ncols=3,
                        tight_layout=True,
                        num="IMAGE_PREPROCESS_METHOD",  # 整幅图的名字，可以使用数字
                        figsize=(6, 4.6),  # 整幅图物理尺寸，宽高单位英寸
                        dpi=200  # 每英寸物理尺寸含有的像素点的数量
                        )

axs[0, 0].imshow(im)
axs[0, 0].set_title("original pic", fontsize='medium')
axs[0, 0].axis("off")
axs[0, 1].imshow(
    cv2.cvtColor(im_gray_left_list[1].astype("uint8"), cv2.COLOR_GRAY2RGB))
axs[0, 1].set_title("1bit-7bit pics", fontsize='medium')
axs[0, 1].axis("off")
# axs[0, 2].imshow(
#     cv2.cvtColor((im_bit_list[0] * 255).astype("uint8"), cv2.COLOR_GRAY2RGB))
axs[0, 2].imshow(im -
                 cv2.cvtColor(im_gray_left_list[1].astype("uint8"),
                              cv2.COLOR_GRAY2RGB))
axs[0, 2].set_title("8bit pic", fontsize='medium')
axs[0, 2].axis("off")

axs[1, 0].imshow(im)
axs[1, 0].set_title("original pic", fontsize='medium')
axs[1, 0].axis("off")
axs[1, 1].imshow(
    cv2.cvtColor(im_add_mask_list[7].astype("uint8"), cv2.COLOR_GRAY2RGB))
axs[1, 1].set_title("2bit-8bit pics", fontsize='medium')
axs[1, 1].axis("off")
axs[1, 2].imshow(
    cv2.cvtColor((im_bit_list[7] * 255).astype("uint8"), cv2.COLOR_GRAY2RGB))
axs[1, 2].set_title("1bit pic", fontsize='medium')
axs[1, 2].axis("off")

plt.savefig('a000_001_output_image/a010_02-06_image_enhancement1.tif')
plt.show()


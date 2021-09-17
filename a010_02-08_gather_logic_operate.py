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

# print(np.mean(body_scan_im.astype("float32")) * 3)

m = np.floor(np.mean(body_scan_im[:, :, 0].astype("float32")))

rows, cols = np.shape(body_scan_im[:, :, 0])

_, im_thresh = cv2.threshold(body_scan_im[:, :, 0], 3 * m, 255,
                             cv2.THRESH_BINARY)
# plt.imshow(im_thresh)
# plt.show()

im_mask = np.ones((rows, cols)) * 3 * m

im = body_scan_im[:, :, 0].astype("float32") * \
     im_thresh.astype("float32") / 255 + \
     im_mask * (1 - im_thresh.astype("float32") / 255)

# plt.imshow(im.astype("uint8"))
# plt.show()

# 确定显示的画幅
fig, axs = plt.subplots(nrows=1,
                        ncols=3,
                        tight_layout=True,
                        num="IMAGE_PREPROCESS_METHOD",  # 整幅图的名字，可以使用数字
                        figsize=(5, 3.2),  # 整幅图物理尺寸，宽高单位英寸
                        dpi=200  # 每英寸物理尺寸含有的像素点的数量
                        )
axs[0].imshow(body_scan_im)
axs[0].set_title("original pic", fontsize='medium')
axs[0].axis("off")
axs[1].imshow(body_scan_im_contrary)
axs[1].set_title("contrary pic", fontsize='medium')
axs[1].axis("off")
axs[2].imshow(cv2.cvtColor(im.astype("uint8"), cv2.COLOR_GRAY2RGB))
axs[2].set_title("output pic", fontsize='medium')
axs[2].axis("off")

plt.savefig("a000_001_output_image/a010_02-08_gather_logic_operate1.tif")
plt.show()

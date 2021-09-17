"""
功能模块：
1. opencv 邻域平均处理，局部平局模糊
2. opencv 添加 padding ，生成 mask
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def gray_im_padding(gray_im, pads=20):
    rows, cols = np.shape(gray_im)
    mask = np.zeros((rows + 2 * pads, cols + 2 * pads))
    mask[pads:rows + pads, pads: cols + pads] = gray_im.astype("float32")

    return mask


# 局部模糊函数
def local_area_fuzz(gray_im, pads=20):
    mask = gray_im_padding(gray_im, pads)

    rows, cols = np.shape(gray_im)
    fuzz_im = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            fuzz_im[i, j] = np.mean(mask[i + pads - pads:i + pads + pads,
                                    j + pads - pads:j + pads + pads])

    # fuzz_im = fuzz_im - np.min(fuzz_im)
    # fuzz_im = fuzz_im / np.max(fuzz_im) * 255

    return fuzz_im


kidney_ori_im = cv2.imread(
    "a000_000_Digital_image_processing_image/DIP3E_Original_Images_CH02/Fig0235(c)(kidney_original).tif")

# plt.imshow(gray_im_padding(kidney_ori_im[:, :, 0], pads=20).astype("uint8"))
# plt.show()
#
# plt.imshow(local_area_fuzz(kidney_ori_im[:, :, 0], pads=20).astype("uint8"))
# plt.show()

kidney_ori_mask = gray_im_padding(kidney_ori_im[:, :, 0], pads=20)
fuzz_im = local_area_fuzz(kidney_ori_im[:, :, 0], pads=20)

# 确定显示的画幅
fig, axs = plt.subplots(nrows=1,
                        ncols=3,
                        tight_layout=True,
                        num="IMAGE_PREPROCESS_METHOD",  # 整幅图的名字，可以使用数字
                        figsize=(5, 2.2),  # 整幅图物理尺寸，宽高单位英寸
                        dpi=200  # 每英寸物理尺寸含有的像素点的数量
                        )
axs[0].imshow(kidney_ori_im)
axs[0].set_title("original pic", fontsize='medium')
axs[0].axis("off")
axs[1].imshow(cv2.cvtColor(kidney_ori_mask.astype("uint8"), cv2.COLOR_GRAY2RGB))
axs[1].set_title("padding pic", fontsize='medium')
axs[1].axis("off")
axs[2].imshow(cv2.cvtColor(fuzz_im.astype("uint8"), cv2.COLOR_GRAY2RGB))
axs[2].set_title("fuzz pic", fontsize='medium')
axs[2].axis("off")
plt.savefig("a000_001_output_image/a010_02-08_z0_area_average.tif")
plt.show()

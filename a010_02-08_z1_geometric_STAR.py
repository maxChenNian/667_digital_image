"""
功能模块：
1. opencv 几何空间变换与图像配准，很有实用性
2. opencv 图像的形态学变换，恒等、尺度、旋转、平移、垂直（偏移）、水平（偏移）变换
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

""" 最近邻插值 Nearest neighbor interpolation """


def nearest_neighbor_interpolation(src_img, dst_height, dst_width):
    src_img_rows, src_img_cols, src_img_channels = np.shape(src_img)
    dst_img_mask = np.zeros(
        (dst_height, dst_width, src_img_channels)).astype("uint8")

    for row in range(dst_height):
        for col in range(dst_width):
            src_x = (row + 1) * (src_img_rows / dst_height) - 2
            src_y = (col + 1) * (src_img_cols / dst_width) - 2
            x = int(np.floor(src_x))
            y = int(np.floor(src_y))
            u = src_x - x
            v = src_y - y
            dst_img_mask[row, col] = (1 - u) * (1 - v) * src_img[x, y] + \
                                     u * (1 - v) * src_img[x + 1, y] + \
                                     (1 - u) * v * src_img[x, y + 1] + \
                                     u * v * src_img[x + 1, y + 1]

    return dst_img_mask


# 图像旋转函数
def im_rotation(im_gray, T_round):
    letter_T_gray = im_gray.astype("float32")
    rows, cols = np.shape(letter_T_gray)
    diagonal_line = int(np.floor(np.sqrt(rows ** 2 + cols ** 2)))

    mask = np.ones((diagonal_line, diagonal_line)) * 0

    for i in range(rows):
        for j in range(cols):
            [x, y, _] = np.dot([i - rows // 2, j - cols // 2, 1], T_round)
            mask[int(x - diagonal_line // 2), int(
                y - diagonal_line // 2)] = letter_T_gray[i, j]

    mask_resize = cv2.resize(mask.astype("uint8"),
                             (diagonal_line, diagonal_line),
                             cv2.INTER_NEAREST)
    # plt.imshow(mask_resize)
    # plt.show()

    return mask_resize.astype("uint8")


letter_T = cv2.imread("a000_000_Digital_image_processing_image/"
                      "DIP3E_Original_Images_CH02/Fig0236(a)(letter_T).tif")
letter_T_gray = letter_T[:, :, 0]

# 建立转换矩阵，旋转的转换矩阵
Theta = np.pi / 6
T_round = [[np.cos(Theta), np.sin(Theta), 0],
           [-np.sin(Theta), np.cos(Theta), 0],
           [0, 0, 1]]
# 生成旋转图像
letter_T_rotation = im_rotation(letter_T[:, :, 0], T_round)

# 确定显示的画幅
fig, axs = plt.subplots(nrows=1,
                        ncols=3,
                        tight_layout=True,
                        num="IMAGE_PREPROCESS_METHOD",  # 整幅图的名字，可以使用数字
                        figsize=(10, 2.2),  # 整幅图物理尺寸，宽高单位英寸
                        dpi=400  # 每英寸物理尺寸含有的像素点的数量
                        )

axs[0].imshow(letter_T)
axs[0].axis("off")
axs[1].imshow(cv2.cvtColor(letter_T_rotation, cv2.COLOR_GRAY2RGB))
axs[1].axis("off")
axs[2].imshow(cv2.cvtColor(letter_T_rotation, cv2.COLOR_GRAY2RGB))
axs[2].axis("off")
plt.show()



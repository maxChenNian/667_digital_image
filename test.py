import cv2
import matplotlib.pyplot as plt
import numpy as np

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


im = cv2.imread('example_02-03_skull.tif')
dst_im = nearest_neighbor_interpolation(src_img=im, dst_height=500,
                                        dst_width=500)
plt.imshow(dst_im)
plt.axis("off")
plt.show()

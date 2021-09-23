"""
功能模块：
1.  几何空间变换 之 恒等变换
2.  基于旋转变换，将图像进行恒等变换
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 前向传递，逆时针旋转图像
def forward_im_rotation(im_gray, degree):
    letter_T_gray = im_gray
    # image size after rotation
    ori_rows, ori_cols = np.shape(letter_T_gray)
    new_rows = np.ceil(np.abs(ori_rows * np.cos(degree)) +
                       np.abs(ori_cols * np.sin(degree)))
    new_cols = np.ceil(np.abs(ori_cols * np.cos(degree)) +
                       np.abs(ori_rows * np.sin(degree)))

    # forward mapping matrices
    m1 = [[1, 0, 0], [0, 1, 0], [-ori_rows / 2, -ori_cols / 2, 1]]
    # 顺时针方向转动
    m2 = [[np.cos(degree), np.sin(degree), 0],
          [-np.sin(degree), np.cos(degree), 0], [0, 0, 1]]
    m3 = [[1, 0, 0], [0, 1, 0], [new_rows / 2, new_cols / 2, 1]]

    ori_letter_T_gray = letter_T_gray.astype("float32")
    new_letter_T_gray = np.ones((int(new_rows), int(new_cols))) * 200

    for i in range(ori_rows):
        for j in range(ori_cols):
            [x, y, _] = np.array([i, j, 1]) @ m1 @ m2 @ m3
            new_letter_T_gray[int(np.floor(x)), int(np.floor(y))] = \
                ori_letter_T_gray[i, j]

    # plt.imshow(new_letter_T_gray.astype("uint8"))
    # plt.show()

    return new_letter_T_gray.astype("uint8")


# 反向传递，逆时针旋转图像，采用最邻近插值算法
def backward_im_rotation(im_gray, degree):
    letter_T_gray = im_gray
    # image size after rotation
    ori_rows, ori_cols = np.shape(letter_T_gray)
    new_rows = np.ceil(np.abs(ori_rows * np.cos(degree)) +
                       np.abs(ori_cols * np.sin(degree)))
    new_cols = np.ceil(np.abs(ori_cols * np.cos(degree)) +
                       np.abs(ori_rows * np.sin(degree)))

    # reverse mapping matrices
    rm1 = [[1, 0, 0], [0, 1, 0], [-new_rows / 2, -new_cols / 2, 1]]
    rm2 = [[np.cos(degree), -np.sin(degree), 0],
           [np.sin(degree), np.cos(degree), 0], [0, 0, 1]]
    rm3 = [[1, 0, 0], [0, 1, 0], [ori_rows / 2, ori_cols / 2, 1]]

    ori_letter_T_gray = letter_T_gray.astype("float32")
    new_letter_T_gray = np.ones((int(new_rows), int(new_cols))) * 200

    for i in range(int(new_rows)):
        for j in range(int(new_cols)):
            [x, y, _] = np.array([i, j, 1]) @ rm1 @ rm2 @ rm3
            if int(np.floor(x)) < 1 or int(np.floor(y)) < 1 or \
                    int(np.floor(x)) >= ori_rows or \
                    int(np.floor(y)) >= ori_cols:
                new_letter_T_gray[i, j] = 200
            else:
                new_letter_T_gray[i, j] = \
                    ori_letter_T_gray[int(np.floor(x)), int(np.floor(y))]

    # plt.imshow(new_letter_T_gray.astype("uint8"))
    # plt.show()

    return new_letter_T_gray.astype("uint8")


# 反向传递，逆时针旋转图像，采用双线性插值
def backward_im_rotation2(im_gray, degree):
    letter_T_gray = im_gray
    # image size after rotation
    ori_rows, ori_cols = np.shape(letter_T_gray)
    new_rows = np.ceil(np.abs(ori_rows * np.cos(degree)) +
                       np.abs(ori_cols * np.sin(degree)))
    new_cols = np.ceil(np.abs(ori_cols * np.cos(degree)) +
                       np.abs(ori_rows * np.sin(degree)))

    # reverse mapping matrices
    rm1 = [[1, 0, 0], [0, 1, 0], [-new_rows / 2, -new_cols / 2, 1]]
    rm2 = [[np.cos(degree), -np.sin(degree), 0],
           [np.sin(degree), np.cos(degree), 0], [0, 0, 1]]
    rm3 = [[1, 0, 0], [0, 1, 0], [ori_rows / 2, ori_cols / 2, 1]]

    ori_letter_T_gray = letter_T_gray.astype("float32")
    new_letter_T_gray = np.ones((int(new_rows), int(new_cols))) * 200

    for i in range(int(new_rows)):
        for j in range(int(new_cols)):
            [x, y, _] = np.array([i, j, 1]) @ rm1 @ rm2 @ rm3
            if int(np.floor(x)) < 1 or int(np.floor(y)) < 1 or \
                    int(np.floor(x)) >= ori_rows or \
                    int(np.floor(y)) >= ori_cols:
                new_letter_T_gray[i, j] = 200
            else:
                # 双线性插值
                left = int(np.floor(y)) - 1
                right = int(np.ceil(y)) - 1
                top = int(np.floor(x)) - 1
                bottom = int(np.ceil(x)) - 1

                a = y - left
                b = x - top
                new_letter_T_gray[i, j] = \
                    (1 - a) * (1 - b) * ori_letter_T_gray[top, left] + \
                    a * (1 - b) * ori_letter_T_gray[top, right] + \
                    (1 - a) * b * ori_letter_T_gray[bottom, left] + \
                    a * b * ori_letter_T_gray[bottom, right]

    # plt.imshow(new_letter_T_gray.astype("uint8"))
    # plt.show()

    return new_letter_T_gray.astype("uint8")


if __name__ == "__main__":
    letter_T = cv2.imread("a000_000_Digital_image_processing_image/"
                          "DIP3E_Original_Images_CH02/Fig0236(a)(letter_T).tif")
    letter_T_gray = letter_T[:, :, 0]

    # 恒等变换即是转动角度为0
    degree = 0

    forward_im = forward_im_rotation(letter_T_gray, degree)

    backward_im_nnp = backward_im_rotation(letter_T_gray, degree)

    backward_im_lp = backward_im_rotation2(letter_T_gray, degree)

    # 确定显示的画幅
    fig, axs = plt.subplots(nrows=1,
                            ncols=4,
                            tight_layout=True,
                            num="IMAGE_PREPROCESS_METHOD",  # 整幅图的名字，可以使用数字
                            figsize=(10, 2.2),  # 整幅图物理尺寸，宽高单位英寸
                            dpi=400  # 每英寸物理尺寸含有的像素点的数量
                            )

    axs[0].imshow(letter_T_gray)
    axs[0].set_title("original image", fontsize='medium')
    axs[0].axis("off")
    axs[1].imshow(forward_im)
    axs[1].set_title("forward rotation", fontsize='medium')
    axs[1].axis("off")
    axs[2].imshow(backward_im_nnp)
    axs[2].set_title("nnp rotation", fontsize='medium')
    axs[2].axis("off")
    axs[3].imshow(backward_im_lp)
    axs[3].set_title("lp rotation", fontsize='medium')
    axs[3].axis("off")
    plt.savefig("a000_001_output_image/a010_02-08_z3_geometric_STAR.jpg")
    plt.show()

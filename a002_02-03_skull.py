""" 改变图像灰度级 """
""" 思路：归一化后乘以灰度级 """

import cv2
import matplotlib.pyplot as plt

im_ori_path = 'a000_000_Digital_image_processing_image/DIP3E_Original_Images_CH02/' \
              'Fig0221(a)(ctskull-256).tif'
im_ori_BGR = cv2.imread(im_ori_path)

gray_levels = [2 ** 8, 2 ** 7, 2 ** 6, 2 ** 5, 2 ** 4, 2 ** 3, 2 ** 2, 2 ** 1]
# print(gray_levels)

# 原始图像灰度级为 0~256
im_gray = im_ori_BGR[:, :, 0]
# cv2.imwrite("skull000_256.bmp", im_gray)

# 原始图像归一化
im_normalized = im_gray.astype("float32") / 255.0

# 确定显示的画幅
fig, axs = plt.subplots(nrows=2,
                        ncols=4,
                        tight_layout=True,
                        num="IMAGE_PREPROCESS_METHOD",  # 整幅图的名字，可以使用数字
                        figsize=(7, 5),  # 整幅图物理尺寸，宽高单位英寸
                        dpi=200  # 每英寸物理尺寸含有的像素点的数量
                        )
# 开始画图
for i, gray_level in enumerate(gray_levels):

    # 改变图像的灰度级
    if 8 - i == 1:
        _, im_gray_out = cv2.threshold(im_gray, 127, 255, cv2.THRESH_BINARY)
    else:
        im_gray_out = (im_normalized * (gray_level - 1)).astype("uint8") * 2 ** i

    im_gray_BGR = cv2.cvtColor(im_gray_out, cv2.COLOR_GRAY2BGR)
    axs[i // 4, i % 4].imshow(im_gray_BGR)
    axs[i // 4, i % 4].set_title("gray_level:" + str(2 ** (8 - i)))
    axs[i // 4, i % 4].axis('off')

    # im_save_name = "example_02-03_skull_" + str("%03d" % gray_level) + ".bmp"
    # cv2.imwrite(im_save_name, im_gray_out)

plt.savefig("example_02-03_skull.tif")
plt.show()




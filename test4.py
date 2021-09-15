import cv2
import matplotlib.pyplot as plt
import numpy as np

im_dollar = cv2.imread(
    "Digital_image_processing_image/DIP3E_Original_Images_CH03/Fig0314(a)(100-dollars).tif")

im_dollar_gray = cv2.cvtColor(im_dollar, cv2.COLOR_BGR2GRAY)

im = im_dollar
rows, cols, _ = np.shape(im)
im_list = []
# im_all = np.zeros((rows, cols))
for i in range(8):
    pass
    a = i // 8
    b = i % 8
    im2 = im[:, :, int(a)]
    rows, cols = np.shape(im2)
    # print(b)
    im3 = np.zeros((rows, cols))
    for ii in range(rows):
        for jj in range(cols):
            # print(bin(im2[ii, jj])[2+b:2+b+1])
            # print(str(np.binary_repr(im2[ii, jj], 8)))
            im3[ii, jj] = int(
                str(np.binary_repr(im2[ii, jj], 8))[b:b + 1]) * 255

    # name = str(i) + ".png"
    # cv2.imwrite(name, im3.astype("uint8"))
    im_list.append(im3.astype("uint8"))
    # im_all = im_all + im3

# 确定显示的画幅
fig, axs = plt.subplots(nrows=3,
                        ncols=3,
                        tight_layout=True,
                        num="IMAGE_PREPROCESS_METHOD",  # 整幅图的名字，可以使用数字
                        figsize=(5, 3),  # 整幅图物理尺寸，宽高单位英寸
                        dpi=200  # 每英寸物理尺寸含有的像素点的数量
                        )
axs[0, 0].imshow(im_dollar)
axs[0, 0].set_title("ori_pic")
axs[0, 0].axis('off')

# 开始画图
for i, im_bit in enumerate(im_list):
    im_bit_BGR = cv2.cvtColor(im_bit, cv2.COLOR_GRAY2BGR)
    # im_save_name = "example_02-02_clock_" + str("%03d" % i) + ".bmp"
    # cv2.imwrite(im_save_name, im_resize)
    axs[(i + 1) // 3, (i + 1) % 3].imshow(im_bit_BGR)
    axs[(i + 1) // 3, (i + 1) % 3].set_title(str(8-i) + "bit")
    axs[(i + 1) // 3, (i + 1) % 3].axis('off')
    # axs[i % 4].set_xticks([])
    # axs[i % 4].set_yticks([])
plt.savefig("example_02-02_clock111.tif")
plt.show()

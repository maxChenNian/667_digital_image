import cv2
import numpy as np
import matplotlib.pyplot as plt

im = cv2.imread(
    "Digital_image_processing_image/DIP3E_Original_Images_CH02/Fig0227(a)(washington_infrared).tif")

im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

im2 = im_gray.astype('float32')

im3 = im2 - im2 % 2
plt.imshow(im3.astype("uint8"))
plt.show()
cv2.imwrite("nnnnn.png", im3.astype("uint8"))

im4 = (im2 - im3) * 255
plt.imshow(im4.astype("uint8"))
plt.show()
cv2.imwrite("nnhao.png", im4.astype("uint8"))
rows, cols = np.shape(im2)

# 确定显示的画幅
fig, axs = plt.subplots(nrows=1,
                        ncols=1,
                        tight_layout=True,
                        num="IMAGE_PREPROCESS_METHOD",  # 整幅图的名字，可以使用数字
                        figsize=(4, 4),  # 整幅图物理尺寸，宽高单位英寸
                        dpi=200  # 每英寸物理尺寸含有的像素点的数量
                        )
axs.imshow(im4.astype("uint8")[0:150, 0:150])
axs.axis("off")
plt.show()

# for i in rows:
#     for j in cols:

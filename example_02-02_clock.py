""" 降低图像空间分辨率 """
""" dpi将像素点个数与现实实际距离联系起来 """
""" 降低图像dpi值，本质上成正比的降低像素点个数 """

import cv2
import matplotlib.pyplot as plt
import numpy as np

im_path = 'a000_001_Digital_image_processing_image/DIP3E_Original_Images_CH02/' \
          'Fig0220(a)(chronometer 3692x2812  2pt25 inch 1250 dpi).tif'
im = cv2.imread(im_path)
# cv2.imwrite('im000_1250dpi.bmp', im)

# im_resize = cv2.resize(im, (675, 887), interpolation=cv2.INTER_NEAREST)
# cv2.imwrite('im001_300dpi.bmp', im_resize)

# im_resize = cv2.resize(im, (338, 444), interpolation=cv2.INTER_NEAREST)
# cv2.imwrite('im002_150dpi.bmp', im_resize)

# im_resize = cv2.resize(im, (162, 213), interpolation=cv2.INTER_NEAREST)
# cv2.imwrite('im003_72dpi.bmp', im_resize)

"""
INTER_NEAREST   最近邻插值

INTER_LINEAR    双线性插值（默认设置）

INTER_AREA      使用像素区域关系进行重采样。

INTER_CUBIC     4x4像素邻域的双三次插值

INTER_LANCZOS4  8x8像素邻域的Lanczos插值
"""

dpi_list = [1250, 300, 150, 72]
im_ori_row, im_ori_col, _ = np.shape(im)

# 确定显示的画幅
fig, axs = plt.subplots(nrows=1,
                        ncols=4,
                        tight_layout=True,
                        num="IMAGE_PREPROCESS_METHOD",  # 整幅图的名字，可以使用数字
                        figsize=(7, 2.5),  # 整幅图物理尺寸，宽高单位英寸
                        dpi=200  # 每英寸物理尺寸含有的像素点的数量
                        )
# 开始画图
for i, dpi in enumerate(dpi_list):
    if i == 0:
        im_resize = im[:, :, :]
    else:
        im_resize = cv2.resize(im, (int(im_ori_col * dpi / 1250),
                                    int(im_ori_row * dpi / 1250)),
                               interpolation=cv2.INTER_NEAREST)
    im_save_name = "example_02-02_clock_" + str("%03d" % i) + ".bmp"
    cv2.imwrite(im_save_name, im_resize)
    axs[i % 4].imshow(im_resize)
    axs[i % 4].set_title(str(dpi) + "dpi")
    # axs[i % 4].axis('off')
    axs[i % 4].set_xticks([])
    axs[i % 4].set_yticks([])
plt.savefig("example_02-02_clock.tif")
plt.show()


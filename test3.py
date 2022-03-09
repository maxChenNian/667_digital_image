import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

with open('0003.txt', 'r') as file:
    a = file.read()
    print(a)

points = a.split("\n")[:241]

x_list = []
y_list = []
for point_str in points:
    [x_str, y_str, _, _] = point_str.split(" ")
    x0 = float(x_str)
    y0 = float(y_str)
    x_list.append(int(x0))
    y_list.append(int(y0))

row_min = np.min(x_list)
row_max = np.max(x_list)

col_min = np.min(y_list)
col_max = np.max(y_list)

mask = np.zeros((row_max - row_min, col_max - col_min))
rows, cols = np.shape(mask)
for ii in range(len(x_list)):

    for i in range(rows):
        for j in range(cols):
            if i == x_list[ii] and j == y_list[ii]:
                # mask[i, j] = random.randint(1, 7) * 20
                mask[i, j] = 255

        # 确定显示的画幅
fig, axs = plt.subplots(nrows=1,
                        ncols=1,
                        tight_layout=True,
                        num="IMAGE_PREPROCESS_METHOD",  # 整幅图的名字，可以使用数字
                        figsize=(3, 2),  # 整幅图物理尺寸，宽高单位英寸
                        dpi=400  # 每英寸物理尺寸含有的像素点的数量
                        )
axs.imshow(mask.T)
plt.show()

aa = mask.T

print()

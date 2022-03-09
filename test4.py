import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cv2

im = cv2.imread("0050.png")
im_resize = cv2.resize(im, (3950, 3500), interpolation=cv2.INTER_CUBIC)
_, im_bin = cv2.threshold(im_resize, 200, 255, cv2.THRESH_BINARY)
# plt.imshow(im_bin.astype("uint8"))
# plt.show()

with open('0003.txt', 'r') as file:
    a = file.read()
    print(a)

points = a.split("\n")[:241]

x_list = []
y_list = []
points2 = []
value_list = []
for point_str in points:
    [x_str, y_str, _, value] = point_str.split(" ")
    # x0 = int(x_str) // 0.9 - 500
    # y0 = int(y_str) // 0.9 + 600
    x0 = float(x_str)
    y0 = float(y_str)
    x_list.append(int(x0))
    y_list.append(int(y0))
    points2.append([int(y0), int(x0)])

    value_list.append(int(value))

# n = 95
# points = np.random.rand(n, 2)  # n是已知点个数
# np.random.seed(2)
# values = np.random.randint(1, 6, n)  # 对应没每个点的值

# 插值的目标
# 注意，这里和普通使用数组的维度、下标不一样，是因为如果可视化的话，imshow坐标轴和一般的不一样
start1 = 0
end1 = 500
step1 = 1
start2 = 0
end2 = 500
step2 = 1

x, y = np.mgrid[
       start1:end1:step1,
       start2:end2:step2]

# grid就是插值结果，你想要的到的区间的每个点数据都在这个grid矩阵里
grid = griddata(points2, np.array(value_list), (x, y), method="cubic", fill_value=0)

# grid = int((grid - np.min(grid)) / (np.max(grid) - np.min(grid)) * 4)
grid = (grid - np.min(grid)) / (np.max(grid) - np.min(grid)) * 4
# aaa = grid * im_bin[:, :, 0] // 255

# 这里通过imshow显示时，坐标思维要按照计算机的来，普通图片是2维数组
# x 是最终结果的第一维，下标是从上到下由零增加
# y 是最终结果的第二维，下标是从左到右由零增加
# plt.subplot(1, 1, 1)
# plt.title("0°")
fig, axs = plt.subplots(nrows=1,
                        ncols=1,
                        tight_layout=True,
                        num="IMAGE_PREPROCESS_METHOD",  # 整幅图的名字，可以使用数字
                        figsize=(7, 5),  # 整幅图物理尺寸，宽高单位英寸
                        dpi=400  # 每英寸物理尺寸含有的像素点的数量
                        )
axs2 = axs.imshow(grid.astype("uint8"), cmap='jet')  # contourf jet gray
fig.colorbar(axs2, ax=axs)
plt.savefig('tunnel_face.jpg')
plt.show()

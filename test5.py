from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt

with open('0000.txt', 'r') as file:
    a = file.read()
    print(a)

points = a.split("\n")[:95]

x_list = []
y_list = []
points2 = []
for point_str in points:
    [x_str, y_str] = point_str.split(" ")
    x0 = int(x_str) // 1
    y0 = int(y_str) // 1
    x_list.append(int(x0))
    y_list.append(int(y0))
    points2.append([int(y0), int(x0)])

row_min = np.min(y_list)
row_max = np.max(y_list)

col_min = np.min(x_list)
col_max = np.max(x_list)

mask = np.zeros((3000, 4050))

ii = 0
np.random.seed(2)

for ii in range(95):
    for i in range(3000):
        for j in range(4050):
            if i == y_list[ii] and j == x_list[ii]:
                mask[i, j] = np.random.randint(1, 6, 1)
                continue

d_row = []
for i in range(3000):
    if np.sum(mask[i, :]) == 0:
        d_row.append(i)

d_col = []
for j in range(4050):
    if np.sum(mask[:, j]) == 0:
        d_col.append(j)

d_row = sorted(d_row, reverse=True)
for d in d_row:
    mask = np.delete(mask, d, 0)

d_col = sorted(d_col, reverse=True)
for d in d_col:
    mask = np.delete(mask, d, 1)

plt.imshow(mask)
plt.show()

fig, axs = plt.subplots(nrows=1,
                        ncols=1,
                        tight_layout=True,
                        num="IMAGE_PREPROCESS_METHOD",  # 整幅图的名字，可以使用数字
                        figsize=(7, 5),  # 整幅图物理尺寸，宽高单位英寸
                        dpi=400  # 每英寸物理尺寸含有的像素点的数量
                        )
axs2 = axs.imshow(mask.astype("uint8"), cmap='jet')  # contourf jet gray
fig.colorbar(axs2, ax=axs)
plt.savefig('tunnel_face.jpg')
plt.show()

# a=np.array(([7,1,2,8],[4,0,3,2],[5,8,3,6],[4,3,2,0]))
# b=[]
#
# for i in range(len(a)):
#     for j in range (len(a[i])):
#         if a[i][j]==0:
#             b.append(i)
#
# print('b=', b)
# b = sorted(b, reverse=True)
# for zero_row in b:
#     a = np.delete(a,zero_row, 0)
#
# print('a=',a)
# print(x)


# row_list = y_list - row_min
# col_list = x_list - col_min
#
# mask = np.zeros((row_max - row_min, col_max - col_min))
# rows = row_max - row_min
# cols = col_max - col_min
#
# np.random.seed(2)
# for i in range(rows):
#     for j in range(cols):
#         mask[i, j] = np.random.randint(1, 6, 1)
#
# plt.imshow(mask)
# plt.show()

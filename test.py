import numpy as np
import cv2
import matplotlib.pyplot as plt


# 双目相机参数，内参参数，利用matlab标定后获取内参参数
class stereoCameras:
    def __init__(self):
        # # 左相机内参数
        # self.LeftIntrinsicMatrix = np.array(
        #     [[1.6076 * 10 ** 3, 0, 2.0312 * 10 ** 3],
        #      [0, 1.6196 * 10 ** 3, 1.1404 * 10 ** 3],
        #      [0, 0, 1]])
        #
        # # 右相机内参数
        # self.RightIntrinsicMatrix = np.array(
        #     [[2.5421 * 10 ** 3, -0.8284, 2.0727 * 10 ** 3],
        #      [0, 2.5190 * 10 ** 3, 932.1327],
        #      [0, 0, 1]])

        # 左相机内参数
        self.LeftIntrinsicMatrix = np.array(
            [[2.1686 * 10 ** 3, 86.6099, 2.4213 * 10 ** 3],
             [0, 2.2751 * 10 ** 3, 974.5184],
             [0, 0, 1]])

        # 右相机内参数
        self.RightIntrinsicMatrix = np.array(
            [[2.5421 * 10 ** 3, -0.8284, 2.0727 * 10 ** 3],
             [0, 2.5190 * 10 ** 3, 932.1327],
             [0, 0, 1]])

        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.LeftDistortion = np.array(
            [-0.2049, -0.0321, 0.0175, -0.0404, 0])
        self.RightDistortion = np.array(
            [-0.2500, -0.2395, 0.0234, -0.0052, 0])

        # 确定相机2相对相机1的旋转和平移矩阵
        self.RotationOfCamera2 = np.array([[0.9266, 0.0032, 0.3760],
                                           [-0.0118, 0.9997, 0.0207],
                                           [-0.3758, -0.0236, 0.9264]])
        self.TranslationOfCamera2 = np.array([-133.4733, -4.4774, 32.0991])


"""
视差以及三维坐标
"""


def getRectifyTransform(height, width, config):
    # 读取矩阵参数
    left_K = config.LeftIntrinsicMatrix
    right_K = config.RightIntrinsicMatrix
    left_distortion = config.LeftDistortion
    right_distortion = config.RightDistortion
    R = config.RotationOfCamera2
    T = config.TranslationOfCamera2

    # 计算校正变换
    if type(height) != "int" or type(width) != "int":
        height = int(height)
        width = int(width)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion,
                                                      right_K, right_distortion,
                                                      (width, height), R, T,
                                                      alpha=0)
    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1,
                                               (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2,
                                               P2, (width, height),
                                               cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q


# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_CUBIC)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)
    return rectifyed_img1, rectifyed_img2


# 视差计算
def sgbm(imgL, imgR):
    # SGBM参数设置
    window_size = 15
    blockSize = 5
    img_channels = 3
    stereo = cv2.StereoSGBM_create(minDisparity=0,
                                   numDisparities=160,
                                   blockSize=blockSize,
                                   P1=8 * img_channels * window_size ** 2,
                                   P2=32 * img_channels * window_size ** 2,
                                   disp12MaxDiff=-1,
                                   preFilterCap=63,
                                   uniquenessRatio=15,
                                   speckleWindowSize=0,
                                   speckleRange=2,
                                   mode=cv2.STEREO_SGBM_MODE_HH)
    # 计算视差图
    disp = stereo.compute(imgL, imgR)
    disp = np.divide(disp.astype(np.float32), 16.)  # 除以16得到真实视差图

    # 转换为单通道图片
    disp = cv2.normalize(disp, disp, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return disp


# 计算三维坐标，并删除错误点
def threeD(disp, Q):
    # 计算像素点的3D坐标（左相机坐标系下）
    points_3d = cv2.reprojectImageTo3D(disp, Q)

    points_3d = points_3d.reshape(points_3d.shape[0] * points_3d.shape[1], 3)

    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2]

    # 选择并删除错误的点
    remove_idx1 = np.where(Z <= 0)
    remove_idx2 = np.where(Z > 15000)
    remove_idx3 = np.where(X > 10000)
    remove_idx4 = np.where(X < -10000)
    remove_idx5 = np.where(Y > 10000)
    remove_idx6 = np.where(Y < -10000)
    remove_idx = np.hstack(
        (remove_idx1[0], remove_idx2[0], remove_idx3[0], remove_idx4[0],
         remove_idx5[0], remove_idx6[0]))

    points_3d = np.delete(points_3d, remove_idx, 0)

    # 计算目标点（这里我选择的是目标区域的中位数，可根据实际情况选取）
    if points_3d.any():
        x = np.median(points_3d[:, 0])
        y = np.median(points_3d[:, 1])
        z = np.median(points_3d[:, 2])
        targetPoint = [x, y, z]
    else:
        targetPoint = [0, 0, -1]  # 无法识别目标区域

    return targetPoint


def update(val=0):
    stereo.setBlockSize(cv2.getTrackbarPos('window_size', 'disparity'))
    stereo.setUniquenessRatio(
        cv2.getTrackbarPos('uniquenessRatio', 'disparity'))
    stereo.setSpeckleWindowSize(
        cv2.getTrackbarPos('speckleWindowSize', 'disparity'))
    stereo.setSpeckleRange(cv2.getTrackbarPos('speckleRange', 'disparity'))
    stereo.setDisp12MaxDiff(cv2.getTrackbarPos('disp12MaxDiff', 'disparity'))

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    cv2.resizeWindow('left', 800, 450)
    cv2.resizeWindow('right', 800, 450)
    cv2.resizeWindow('disparity', 800, 450)

    # cv2.imshow('left', imgL)
    # cv2.imshow('right', imgR)
    # cv2.imshow('disparity', (disp - min_disp) / num_disp)
    cv2.imshow('left',
               cv2.resize(imgL, (800, 450), interpolation=cv2.INTER_CUBIC))
    cv2.imshow('right',
               cv2.resize(imgR, (800, 450), interpolation=cv2.INTER_CUBIC))
    cv2.imshow('disparity', cv2.resize((disp - min_disp) / num_disp, (800, 450),
                                       interpolation=cv2.INTER_CUBIC))


if __name__ == "__main__":
    # aa = cv2.Rodrigues(np.array([-0.5178, 0.0303, 2.9791]))
    # print()

    # imgL = cv2.imread("/home/cdjs/PythonProject/TensorFlow/cameras/left/01.jpg")
    # imgR = cv2.imread(
    #     "/home/cdjs/PythonProject/TensorFlow/cameras/right/01.jpg")

    imgL = cv2.imread(r"C:/Users/JinSui001/Desktop/cameras/left/01.jpg")
    imgR = cv2.imread(r"C:/Users/JinSui001/Desktop/cameras/right/01.jpg")

    height, width = imgL.shape[0:2]
    # 读取相机内参和外参
    config = stereoCameras()

    print()

    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)
    iml_rectified, imr_rectified = rectifyImage(imgL, imgR, map1x, map1y, map2x,
                                                map2y)

    # plt.imshow(imgL)
    # plt.show()
    #
    # plt.imshow(imgR)
    # plt.show()

    # plt.imshow(iml_rectified)
    # plt.show()
    #
    # plt.imshow(imr_rectified)
    # plt.show()

    # iml_rectified2 = iml_rectified[:, :, :]
    # imr_rectified2 = imr_rectified[:, :, :]
    # interval_num = int(height // 11)
    # rows_start = 0
    # for i in range(10):
    #     rows_start = rows_start + interval_num
    #     # 起点和终点的坐标
    #     ptStart = (0, rows_start)
    #     ptEnd = (width, rows_start)
    #     point_color = (0, 255, 0)  # BGR
    #     thickness = 2
    #     lineType = 4
    #     cv2.line(iml_rectified2, ptStart, ptEnd, point_color, thickness,
    #              lineType)
    #     cv2.line(imr_rectified2, ptStart, ptEnd, point_color, thickness,
    #              lineType)

    # cv2.imwrite("im_rectified1.jpg", np.hstack((iml_rectified2, imr_rectified2)))

    window_size = 5
    min_disp = 16
    num_disp = 192 - min_disp
    blockSize = window_size
    uniquenessRatio = 1
    speckleRange = 12
    speckleWindowSize = 3
    disp12MaxDiff = 200
    P1 = 600
    P2 = 2400

    # imgL = cv2.imread('depth1.jpg')
    # imgR = cv2.imread('depth2.jpg')
    imgL = iml_rectified
    imgR = imr_rectified

    cv2.namedWindow('disparity')
    cv2.createTrackbar('speckleRange', 'disparity', speckleRange, 50, update)
    cv2.createTrackbar('window_size', 'disparity', window_size, 21, update)
    cv2.createTrackbar('speckleWindowSize', 'disparity', speckleWindowSize, 200,
                       update)
    cv2.createTrackbar('uniquenessRatio', 'disparity', uniquenessRatio, 50,
                       update)
    cv2.createTrackbar('disp12MaxDiff', 'disparity', disp12MaxDiff, 250, update)

    '''
    cv2.createStereoSGBM(minDisparity, numDisparities, blockSize[, P1[, P2[, disp12MaxDiff[, preFilterCap[, uniquenessRatio[, speckleWindowSize[, speckleRange[, mode]]]]]]]]) → retval
    Parameters:
        minDisparity – Minimum possible disparity value. Normally, it is zero but sometimes rectification algorithms can shift images, so this parameter needs to be adjusted accordingly.
        numDisparities – Maximum disparity minus minimum disparity. The value is always greater than zero. In the current implementation, this parameter must be divisible by 16.
        blockSize – Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
        P1 – The first parameter controlling the disparity smoothness. See below.
        P2 – The second parameter controlling the disparity smoothness. The larger the values are, the smoother the disparity is. P1 is the penalty on the disparity change by plus or minus 1 between neighbor pixels. P2 is the penalty on the disparity change by more than 1 between neighbor pixels. The algorithm requires P2 > P1 . See stereo_match.cpp sample where some reasonably good P1 and P2 values are shown (like 8*number_of_image_channels*SADWindowSize*SADWindowSize and 32*number_of_image_channels*SADWindowSize*SADWindowSize , respectively).
        disp12MaxDiff – Maximum allowed difference (in integer pixel units) in the left-right disparity check. Set it to a non-positive value to disable the check.
        preFilterCap – Truncation value for the prefiltered image pixels. The algorithm first computes x-derivative at each pixel and clips its value by [-preFilterCap, preFilterCap] interval. The result values are passed to the Birchfield-Tomasi pixel cost function.
        uniquenessRatio – Margin in percentage by which the best (minimum) computed cost function value should “win” the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enough.
        speckleWindowSize – Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        speckleRange – Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
        mode – Set it to StereoSGBM::MODE_HH to run the full-scale two-pass dynamic programming algorithm. It will consume O(W*H*numDisparities) bytes, which is large for 640x480 stereo and huge for HD-size pictures. By default, it is set to false .
    '''

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        uniquenessRatio=uniquenessRatio,
        speckleRange=speckleRange,
        speckleWindowSize=speckleWindowSize,
        disp12MaxDiff=disp12MaxDiff,
        P1=P1,
        P2=P2
    )
    update()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # plt.imshow(np.hstack((iml_rectified, imr_rectified)))
    # plt.show()
    #
    # print()
    #
    # disp = sgbm(iml_rectified, imr_rectified)
    # # cv2.imshow("disp", disp)
    # plt.imshow(disp)
    # plt.show()
    # print(np.max(disp))

    # target_point = threeD(disp, Q)  # 计算目标点的3D坐标（左相机坐标系下）
    # print(target_point)

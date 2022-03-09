import cv2
import numpy as np
import matplotlib.pyplot as plt

imgL = cv2.imread(r"C:/Users/JinSui001/Desktop/cameras/left/01.jpg")
imgR = cv2.imread(r"C:/Users/JinSui001/Desktop/cameras/right/01.jpg")
# imgL = cv2.imread(r"11.png")
# imgR = cv2.imread(r"22.png")

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

    # plt.imshow(cv2.resize(imgL, (450, 800), interpolation=cv2.INTER_CUBIC))
    # plt.show()

    cv2.imshow('left',
               cv2.resize(imgL, (800, 450), interpolation=cv2.INTER_CUBIC))
    cv2.imshow('right', cv2.resize(imgR, (800, 450), interpolation=cv2.INTER_CUBIC))
    cv2.imshow('disparity', cv2.resize((disp - min_disp) / num_disp, (800, 450), interpolation=cv2.INTER_CUBIC))


cv2.namedWindow('disparity')
cv2.createTrackbar('speckleRange', 'disparity', speckleRange, 50, update)
cv2.createTrackbar('window_size', 'disparity', window_size, 21, update)
cv2.createTrackbar('speckleWindowSize', 'disparity', speckleWindowSize, 200,
                   update)
cv2.createTrackbar('uniquenessRatio', 'disparity', uniquenessRatio, 50,
                   update)
cv2.createTrackbar('disp12MaxDiff', 'disparity', disp12MaxDiff, 250, update)

cv2.resizeWindow('left', 800, 450)
cv2.resizeWindow('right', 800, 450)
cv2.resizeWindow('disparity', 800, 450)

update()
cv2.waitKey(0)
cv2.destroyAllWindows()

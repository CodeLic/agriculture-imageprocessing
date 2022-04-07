import cv2
import numpy as np
import matplotlib.pyplot as plt
from skvideo.io import vread
import moviepy.editor as mpy
from tqdm import tqdm
from mpl_toolkits.mplot3d import axes3d, Axes3D

XYZ = []
RPY = []
V_x = []
V_y = []
V_z = []

# cv2.arucoのインスタンス化
ARUCO = cv2.aruco

# 4ブロック×4ブロックを50個使用可能
aruco_dict = ARUCO.getPredefinedDictionary(ARUCO.DICT_4X4_50)

for frame in vid[:500:25]:  # 全部処理すると重いので…
    frame = frame[...,::-1]  # BGR2RGB
    corners, ids, _ = ARUCO.detectMarkers(frame, aruco_dict)

    if len(corners) == 0:
        continue

    # mtx: カメラの内部パラメータ, dist: 歪み係数, marker_length: マーカーの長さ(単位: m)
    rvec, tvec, _ = ARUCO.estimatePoseSingleMarkers(corners, marker_length, mtx, dist)

    R = cv2.Rodrigues(rvec)[0]  # 回転ベクトル -> 回転行列
    R_T = R.T
    T = tvec[0].T

    xyz = np.dot(R_T, - T).squeeze()
    XYZ.append(xyz)

    rpy = np.deg2rad(cv2.RQDecomp3x3(R_T)[0])
    RPY.append(rpy)

    V_x.append(np.dot(R_T, np.array([1,0,0])))
    V_y.append(np.dot(R_T, np.array([0,1,0])))
    V_z.append(np.dot(R_T, np.array([0,0,1])))

    # ---- 描画
    ARUCO.drawDetectedMarkers(frame, corners, ids, (0,255,255))
    ARUCO.drawAxis(frame, mtx, dist, rvec, tvec, marker_length/2)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    # ----

cv2.destroyAllWindows()
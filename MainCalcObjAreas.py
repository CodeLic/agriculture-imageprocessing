import os
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2

def resizeImage(images):
    """
    (1)
    画素数を４分の１にエンコード

    Args:
        images: 画像パス
        img: 入力したエンコードするBGR画像
        height: 画像の高さ
        width: 画像の幅

    Returns:
        imgResize: リサイズした画像
    """

    # フォルダのjpgを全てリサイズする
    for filename in images:

        # 画像読み込み
        img = cv2.imread(filename)

        # 高さを定義する
        height = img.shape[0]

        # 幅を定義
        width = img.shape[1]

        # リサイズ処理(高さと幅の画素数を4分の1にする)
        imgResize = cv2.resize(img, (int(width/4), int(height/4)) )

        # リサイズした画像を指定したパスに保存(保存用ディレクトリを用意)
        cv2.imwrite('/home/pi/agriculture-imageprocessing/CVCameraCalibrateImages/D_ResizeChessPatternImages/' + 'ReSize' +  str(os.path.basename(filename)), imgResize)

    # エンコード完了の通知
    print('ImageEcodeComplete')

    return imgResize

def calibrateCamera(chessimages):
    """
    (2)
    取得したチェス盤画像を用いる
    画像からカメラキャリブレーションを行い内部パラメータと歪み係数を出力

    Args:
        CHECKERBOARD: チェス盤面のマス目を定義
        objpoints: 各チェックボードから得た3Dベクトルを格納する配列の宣言
        imgpoints: 各チェックボードから得た2Dベクトルを格納する配列の宣言

    Returns:
        mtx: キャリブレーションで取得した内部パラメータ
        dist: キャリブレーションで取得した歪み係数

    """

    # チェス盤の盤面数を設定
    CHECKERBOARD = (7,10)

    # cv2.TERM_CRITERIA_EPS: 指定された精度(epsilon)に到達したら繰り返し計算を終了する
    # cv2.TERM_CRITERIA_MAX_ITER: 指定された繰り返し回数(max_iter)に到達したら繰り返し計算を終了する
    # cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER : 上記のどちらかの条件が満たされた時に繰り返し計算を終了する
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 各チェックボードから得た3Dベクトルを格納する配列の宣言
    objpoints = []

    # 各チェックボードから得た2Dベクトルを格納する配列の宣言
    imgpoints = []


    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    for filepath in chessimages:
        # 画像の取得
        img = cv2.imread(filepath)

        # グレースケール変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 画像からチェスボードの角を取得する。画像から角がみつかればretにTrueを宣言する
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)

            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

            # 取得した２Dベクトルを追加していく
            imgpoints.append(corners2)

            # 角を画像に描写
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

    """
    Performing camera calibration by
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the
    detected corners (imgpoints)出力
    """

    # 算出したベクトル配列からカメラのキャリブレーションの実行、内部パラメータ、歪み係数のみ使用する
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    # 内部パラメータの表示
    print("Camera matrix : \n")
    print(mtx)

    # 歪み係数の表示
    print("dist : \n")
    print(dist)

    # 内部パラメータと歪み係数を返す
    return mtx, dist

def tranceformingImage(images, mtx, dist):
    """
    (3)
    (2)で取得した内部パラメータ、歪み係数を用いて物体撮影画像を広角補正
    Args:
        images: MAUE入力した物体画像
        mtx: (2)で取得した内部パラメータ
        dist: (2)で取得した歪み係数
        h: 画像の高さ
        w: 画像の幅

    Returns:
        dst: 補正処理を行った画像

    """

    # Using the derived camera parameters to undistort the image
    # for filepath in images:

        # 画像の取得
        # img = cv2.imread(filepath)

    # リサイズした画像の高さ、幅の取得
    h,w = images.shape[:2]

    # Refining the camera matrix using parameters obtained by calibration
    # ROI:Region Of Interest(対象領域)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # Method 1 to undistort the image
    dst = cv2.undistort(images, mtx, dist, None, newcameramtx)

    # 補正した画像を返す
    return dst

def calcObjectArea(image):
    """aruco
    (4)
    物体の実面積を求める
    Args:

    Returns:
        objAreas: 物体の実面積(cm2)

    """
    # ランドマークの頂点座標を格納する配列の宣言
    corners2 = [np.empty((1,4,2))]*4

    # 4ブロック×4ブロックを50個使用可能
    p_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # ランドマーク検出
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, p_dict)

    # 検出結果を元画像に描写
    img_marked = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)

    # BGR画像からRGB画像に変換
    img_marked_rgb = cv2.cvtColor(img_marked,cv2.COLOR_BGR2RGB)

    # 各ランドマークに頂点座標を格納する
    for i,c in zip(ids.ravel(), corners):
        corners2[i] = c.copy()

    # 物体測定エリアを設定するランキャリブレーションドマークの頂点を設定する配列の宣言
    m = np.empty((4,2))

    # 頂点座標の格納
    m[0] = corners2[0][0][2]
    m[1] = corners2[1][0][3]
    m[2] = corners2[2][0][0]
    m[3] = corners2[3][0][1]
    marker_coordinates = np.float32(m)

    # フォーカスした画像の高さを計算
    h1 = ((corners2[0][0][2][0]-corners2[3][0][1][0])**2+(corners2[0][0][2][1]-corners2[3][0][1][1])**2)**0.5
    h2 = ((corners2[2][0][0][0]-corners2[1][0][3][0])**2+(corners2[2][0][0][1]-corners2[1][0][3][1])**2)**0.5
    hhh = (h1+h2)/2
    # print(hhh)

    # フォーカスした幅を計算
    w1 = ((corners2[0][0][2][0]-corners2[1][0][3][0])**2+(corners2[0][0][2][1]-corners2[1][0][3][1])**2)**0.5
    w2 = ((corners2[2][0][0][0]-corners2[3][0][1][0])**2+(corners2[2][0][0][1]-corners2[3][0][1][1])**2)**0.5
    www = (w1+w2)/2
    # print(www)

    # 変形後画像サイズ 縦横比は上記で計算したピクセルを参考にする。
    width, height = (int(www), int(hhh))

    # 実際に表示する画像の座標を格納メイン処理
    true_coordinates = np.float32([[0,0],[width,0],[width,height],[0,height]])

    # 変換率の取得
    trans_mat = cv2.getPerspectiveTransform(marker_coordinates,true_coordinates)

    # 台形補正
    img_trans = cv2.warpPerspective(image,trans_mat,(width, height))

    # BGRからRGBに変換
    img_trans_rgb = cv2.cvtColor(img_trans,cv2.COLOR_BGR2RGB)

    # グレースケール変換メイン処理
    img_trans_gray = cv2.cvtColor(img_trans,cv2.COLOR_BGR2GRAY)
    img_trans_gray_rgb = cv2.cvtColor(img_trans_gray,cv2.COLOR_GRAY2RGB)

    # 画像の二値化
    _,p_binary = cv2.threshold(img_trans_gray,190,255,cv2.THRESH_BINARY)
    p_binary_rgb = cv2.cvtColor(p_binary,cv2.COLOR_GRAY2RGB)
    p_binary = cv2.bitwise_not(p_binary)
    plt.imshow(p_binary_rgb)

    p_contours, _ = cv2.findContours(p_binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    p_and_contours = np.copy(img_trans)
    min_p_area = 60
    large_contours = [cnt for cnt in p_contours if cv2.contourArea(cnt)>min_p_area]
    draw_p = cv2.drawContours(p_and_contours,large_contours,-1,(0,255,0))

    sarea = 0
    for i, cnts in enumerate(large_contours):
        # 輪郭の面積を計算する。
        area = cv2.contourArea(cnts)
        sarea = sarea+area

    # 実面積測定計算処理
    objAreas = sarea * (12.6 * 21.2) / (www * hhh)

    return objAreas

def mainCalcObjAreas():
    """
    メイン処理部分
    Args:
        CHESSBOARDIMAGES: キャリブレーション用のチェスボード画像
        OBJECTIMAGEAS: 面積推定したい物体画像
        RESIZECBIMAGES: リサイズしたチェス画像
        RESIZEOBJECTIMAGES: リサイズした物体画像

    """
    # チェスボード画像パス
    CHESSBOARDIMAGES = glob.glob('/home/pi/agriculture-imageprocessing/CVCameraCalibrateImages/D_ChessPatternImages/*.jpg')

    # エンコードしたチェス盤画像のパス
    RESIZECBIMAGES = glob.glob('/home/pi/agriculture-imageprocessing/CVCameraCalibrateImages/D_ResizeChessPatternImages/*.jpg')

    # 測定したい物体(パスを指定する)
    OBJECTIMAGEAS = glob.glob('/home/pi/agriculture-imageprocessing/CVCameraCalibrateImages/ElemImage/GUM_NANAME.jpg')
    # ##########################################################################################################################

    # (1)チェス盤画像エンコード
    resizeImage(CHESSBOARDIMAGES)

    # (1)物体画像エンコード
    RESIZEOBJECTIMAGES = resizeImage(OBJECTIMAGEAS)

    # (2)カメラキャリブレーション
    MATRIX, DISTORTION = calibrateCamera(RESIZECBIMAGES)

    # (3)物体画像の広角補正
    CALIBRATEIMAGE = tranceformingImage(RESIZEOBJECTIMAGES, MATRIX, DISTORTION)

    # (4)物体実面積測定処理
    OBJECTAREAS = calcObjectArea(CALIBRATEIMAGE)

    # 検出した総面積の表示
    print("総面積={:.2f}cm2".format(OBJECTAREAS))

if __name__ == '__main__':
    mainCalcObjAreas()
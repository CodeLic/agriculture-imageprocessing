# ライブラリimport
from urllib import response
import cv2
import requests
import numpy as np

# arucoのインスタンス
ARUCO=cv2.aruco

# 判定用
DICT = False

# 画像のURL
URL = "http://192.168.2.164:8080/?action=snapshot"

# 4ブロック×4ブロックを50個使用可能
p_dict = ARUCO.getPredefinedDictionary(ARUCO.DICT_4X4_50)

# 撮影した画像の受け取り
response = requests.get(URL)
img = response.content

# 取得確認表示
image = None
if(img):
    print("GET_PICS")
    image = np.asarray(bytearray(img), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    cv2.imshow('test', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Undifind")

#with open("GetPic.jpg", "wb") as snapshot:
#    snapshot.write(img)

# imgに撮影した画像を格納していきましょう
# image = cv2.imread('/home/pi/agriculture-imageprocessing/GetPic.jpg')

# arucoマーカー検出
corners, ids, rejectedImgPoints = ARUCO.detectMarkers(image, p_dict)

#if (rejectedImgPoints):
#    print(ids)
#    print(corners)
#    print(rejectedImgPoints)
#    DICT = True
#    print(DICT)
#else:
#    DICT = False
#    print(DICT)
if ids != None:
    print('succeed on detecting')
    print(ids)
    print(corners)

else:
    print('fail')

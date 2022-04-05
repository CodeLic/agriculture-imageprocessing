# ライブラリimport
from urllib import response
import cv2
import requests

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
if(img):
    print("GET_PICS")
else:
    print("Undifind")

# requestsで取得した画像をロカール保存
with open("GetPic.jpg", "wb") as snapshot:
    snapshot.write(img)

# 保存された画像を取得
image = cv2.imread("/home/pi/agriculture-imageprocessing/GetPic.jpg")

# arucoマーカー検出
corners, ids, rejectedImgPoints = ARUCO.detectMarkers(image, p_dict)

# 検出してマーカのポイントを取得できた場合True判定
if (rejectedImgPoints):
    DICT = True
    print(DICT)
else:
    DICT = False
    print(DICT)

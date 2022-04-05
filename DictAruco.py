#ライブラリimport
from urllib import response
import cv2
import requests

# arucoのインスタンス
aruco=cv2.aruco

# 判定用
DICT = False

# 画像のURL
URL = "http://192.168.2.164:8080/?action=snapshot"

# 4ブロック×4ブロックを50個使用可能
p_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# 撮影した画像の受け取り
response = requests.get(URL)
img = response.content

# 取得確認表示
if(img):
    print("GET_PICS")
else:
    print("Undifind")

with open("GetPic.jpg", "wb") as snapshot:
    snapshot.write(img)

# imgに撮影した画像を格納していきましょう
# img = cv2.imread('/content/drive/MyDrive/IMG_0086.jpg')

# False用テスト
# img = cv2.imread('/content/drive/MyDrive/CVCameraCalibrateImages/ElemImage/SAMPLE_NEAR.jpg')

# 検出
corners, ids, rejectedImgPoints = aruco.detectMarkers(snapshot, p_dict)

if (rejectedImgPoints):
    DICT = True
    print(DICT)
else:
    DICT = False
    print(DICT)

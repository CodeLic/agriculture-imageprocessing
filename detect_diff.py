from DictAruco import getCorners

def detect_diff():
  height = 720 # px
  center = height / 2
  threshold = 10 # px
  corners = getCorners()
  if len(corners) != 0:
    # マーカーの取得成功
    left_bottom = corners[0]
    left_top = corners[1]
    right_top = corners[2]
    right_bottom = corners[3]

    # マーカーは水平と仮定する
    # 左側だけで検査すればよい
    # 右側でも検査して結果がある程度以上ずれればマーカーが傾いていることも検知できる
      
    # 中点取得
    marker_center = left_top[1] + left_bottom[1] / 2
    diff = center - marker_center
    if abs(diff) > threshold:
      print('ずれています')
    else:
      print('ずれていません')
    print(diff)
    return marker_center - center

if __name__ == '__main__':
  detect_diff()
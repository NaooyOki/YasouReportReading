import cv2
import numpy as np
import math
import json
import glob
import os


def debugTmpImgRemove():
    """
    デバッグモードの場合に、tmpフォルダ以下のイメージファイルを一括削除する
    """
    if (__debug__):
        file_list = glob.glob("./tmp/*.jpg")
        print(f"remove tmp files")
        for file in file_list:
            # print(f"remove: {file}")
            os.remove(file)


def debugImgWrite(img, step:str="", detail:str=""):
    """
    デバッグモードの場合に、指定したイメージをファイルに出力する
    """
    if (__debug__):
        cv2.imwrite(f"./tmp/{step}_{detail}.jpg", img)


def trimPosList(lst:list(), skip:int = 2) -> list():
    """
    隣接する直線を一つに束ねる。表の罫線は太いため、直線検出では複数の直線として得られるため。
    @param list:list(int)  線の位置
    @return list:list(start:int, end:int)  束ねた線の位置(始点と終点のタプル) 
    """
    trimed = []
    prev = lst[0]
    start = prev
    for pos in lst:
        if ((pos - prev) > skip):
            trimed.append((start, prev))
            start = pos
        prev = pos
    trimed.append((start, prev))

    return(trimed)


def trimAsRectangle(img, top_left, top_right, bot_left, bot_right):
    """
    指定した4点の矩形で、画像イメージをトリミングする。台形補正や横写しの画像を長方形に補正する
    @param img  処理対象の画像イメージ
    @param top_left:(x, y)   トリミング対象の左上の位置
    @param top_right:(x, y)  トリミング対象の右上の位置
    @param bot_left:(x, y)   トリミング対象の左下の位置
    @param bot_right:(x, y)  トリミング対象の右下の位置
    """

    w_ratio = 1.0   # 縦横比 1.0以上は横長

    #　幅取得
    o_width = np.linalg.norm(top_right - top_left)
    o_width = math.floor(o_width * w_ratio)
    
    #　高さ取得
    o_height = np.linalg.norm(bot_left - top_left)
    o_height = math.floor(o_height)
    
    # 変換前の4点の座標
    src = np.float32([top_left, top_right, bot_left, bot_right])
    
    # 変換後の4点の座標
    dst = np.float32([[0, 0],[o_width, 0],[0, o_height],[o_width, o_height]])
    
    # 変換行列
    M = cv2.getPerspectiveTransform(src, dst)
    
    # 射影変換・透視変換する
    output = cv2.warpPerspective(img, M,(o_width, o_height))
    
    return(output)

def calcRectEdge(contour):
    """
    輪郭線のリストから、左上、右上、左下、右下の位置を求める
    @param contour:list(list(position))
    @return (左上座標, 右上座標, 左下座標, 右下座標)
    """
    # 四隅を求める
    p1 = contour[0][0]
    p2 = contour[0][0]
    p3 = contour[0][0]
    p4 = contour[0][0]
    for point in contour:
        if (p1[0]+p1[1] > point[0][0]+point[0][1]):
            p1 = point[0]
        if (p2[0]-p2[1] < point[0][0]-point[0][1]):
            p2 = point[0]
        if (p3[0]-p3[1] > point[0][0]-point[0][1]):
            p3 = point[0]
        if (p4[0]+p4[1] < point[0][0]+point[0][1]):
            p4 = point[0]
    return (p1, p2, p3, p4)

def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    """
    指定された画像イメージを縦に連結する。最も横幅が狭い画像にリサイズして連結する。
    @param im_list:list(画像イメージ)
    @return 画像イメージ
    """
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)
            
import cv2
import numpy as np
import math
import json
import glob
import os
import re


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





def trimAsRectangle(img, top_left, top_right, bot_left, bot_right):
    """
    指定した4点の矩形で、画像イメージをトリミングする。台形補正や横写しの画像を長方形に補正する
    @param img  処理対象の画像イメージ
    @param top_left:(x, y)   トリミング対象の左上の位置
    @param top_right:(x, y)  トリミング対象の右上の位置
    @param bot_left:(x, y)   トリミング対象の左下の位置
    @param bot_right:(x, y)  トリミング対象の右下の位置
    """

    ratio = 1.0   # 縦横比 1.0以上は横長

    #　変換後の区画の幅と高さを計算する。左上を基準にする。
    width = math.floor(np.linalg.norm(top_right - top_left) * ratio)
    height = math.floor(np.linalg.norm(bot_left - top_left) * ratio)
    
    # 変換前の4点の座標
    src = np.float32([top_left, top_right, bot_left, bot_right])
    
    # 変換後の4点の座標
    dst = np.float32([[0, 0],[width, 0],[0, height],[width, height]])
    
    # 変換行列
    M = cv2.getPerspectiveTransform(src, dst)
    
    # 射影変換・透視変換する
    output = cv2.warpPerspective(img, M,(width, height))
    
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

def get_match_value(text:str, pattern, error_value="?") -> str:
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return error_value
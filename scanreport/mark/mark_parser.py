import cv2
import numpy as np
import math
import json
import glob
import csv
import os
import sys
import inspect
from dataclasses import dataclass, field, asdict
from typing import ClassVar
from dataclasses_json import dataclass_json, config
from marshmallow import Schema, fields

from ..util import *
# import scanreport.util.utility as util


class MarkImageParser():
    def __init__(self) -> None:
        self.maskImage = None

    def readMarkBase(self, base_img, verify_num:int=0):
        """
        蕾 花 実 などが書かれた下地を元にマスクを作る
        
        Params
            base_img: cell_imgのベースになるイメージ。下地だけが書かれており、cell_imgとの差分で、〇を検出する。
        
        Return
            成否
        """
        # 四隅をマスクする
        maskH = base_img.shape[0]
        maskW = base_img.shape[1]
        maskWH = min(maskH, maskW)

        # 四隅をマスクする
        base = base_img.copy()
        base[0:5, 0:base.shape[1]] = 0
        base[base.shape[0]-5:base.shape[0], 0:base.shape[1]] = 0
        base[0:base.shape[0], 0:5] = 0
        base[0:base.shape[0], base.shape[1]-5:base.shape[1]] = 0
        
        # マスクしやすくするために、下地をぼかす
        debugImgWrite(base, "step5", "mask0_raw")
        mask = cv2.blur(base,(5,5))
        debugImgWrite(mask, "step5", "mask1_blur")

        # マスクの下地を作る
        mask0 = np.zeros((base_img.shape[0], base_img.shape[1]), np.uint8)
        mask0[0:5, 0:mask0.shape[1]] = 255
        mask0[mask0.shape[0]-5:mask0.shape[0], 0:mask0.shape[1]] = 255
        mask0[0:mask0.shape[0], 0:5] = 255
        mask0[0:mask0.shape[0], mask0.shape[1]-5:mask0.shape[1]] = 255
        debugImgWrite(mask0, "step5", "mask2_mask0")

        # 下地の中から、文字と思われる塊を検出する "蕾  花  実  胞" なら4つの塊
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        symbols = []
        detected = []
        minWH = min(mask0.shape[0], mask0.shape[1])/3    # 幅か高さの狭いほうの1/3以上の領域だけをマークの下地とみなす
        for j, contour in enumerate(contours):
            size = cv2.contourArea(contour)
            if (size > minWH * minWH):
                x,y,w,h = cv2.boundingRect(contour)
                if ((w > minWH) and (h > minWH)):
                    symbols.append((x,y,w,h))
                    detected.append(False)
                    cv2.rectangle(mask0, (x, y), (x+w, y+h), 255, -1)
        # 左から順にソートする
        symbols = sorted(symbols, key=lambda s: s[0])
        debugImgWrite(mask0, "step5", "mask2_mask0")
        self.maskImage = mask0
        self.symbols = symbols

        # 〇を検出する領域を計算する
        self.symbolAreas = []
        for symbol in symbols:
            centerX = symbol[0] + symbol[2]/2
            centerY = symbol[1] + symbol[3]/2
            symbolArea = (max(centerX - maskWH/2, 0), max(centerY - maskWH/2, 0), maskWH, maskWH)
            self.symbolAreas.append(symbolArea)

        if (verify_num > 0):
            assert len(symbols) == verify_num, f"マスクの数({len(symbols)})が期待({verify_num})と一致しません"




    def detectMarks(self, cell_img):
        """
        蕾 花 実 などの下地を囲む、〇を検出する
        
        Params
            cell_img: 検出対象のイメージ。0~複数の〇で囲まれていることを想定している。
        
        Return
            検出有無(True/Falase)のリスト
                True: マークで囲まれている
                False: 何も囲まれていない(Baseと同じ)
        """

        detected1 = [0] * len(self.symbols)
        detected2 = [0] * len(self.symbols)


        # イメージサイズをもとに、検出する図形の最小値を決めておく
        img_width = cell_img.shape[1]
        img_height = cell_img.shape[0]
        min_width = img_width * 0.3 / len(self.symbols)
        min_height = img_height * 0.3
        

        # 検出対象イメージを下地でマスクし、下地以外のイメージを摘出する
        mask2 = cv2.resize(cv2.bitwise_not(self.maskImage), (img_width, img_height))
        marks = cv2.bitwise_and(cell_img, mask2)
        debugImgWrite(marks, "step5", "marks0_raw")
        marks2 = cv2.blur(marks,(5,5))
        debugImgWrite(marks2, "step5", "marks1_blur")

        # マークを検出する
        contours, hierarchy = cv2.findContours(marks2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(contours):
            # 輪郭領域を求めて、輪郭が囲むシンボルの位置を計算する
            (x, y, w, h) = cv2.boundingRect(contour)
            if ((w < min_width) and (h < min_height)):
                continue

            # 軌跡を囲む矩形内に、シンボルの中心が含まれていれば、〇とみなす
            for j, bound in enumerate(self.symbols):
                # 中心点を含む領域なら、〇とみなす
                if ((x < bound[0]+bound[2]/2 < x+w) and (y < bound[1]+bound[3]/2 < y+h)):
                    # print(f"detected mark: position={j}")
                    detected1[j] = 1
                    cv2.rectangle(marks, (x, y), (x+w, y+h), 255, 2)

            # 軌跡の重心がシンボルの周辺にあれば、〇とみなす
            mu = cv2.moments(contour)
            mx, my = int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])        
            for j, bound in enumerate(self.symbolAreas):
                if ((bound[0] < mx < bound[0]+bound[2]) and (bound[1] < my < bound[1]+bound[3])):
                    # print(f"detected mark: position={j}")
                    detected2[j] = 1
                    cv2.rectangle(marks, (x, y), (x+w, y+h), 255, 2)


        detected = np.add(detected1, detected2)
        if (detected1 != detected2):
            print(f"シンボルの検出結果が違います。detected={detected1}, detected2={detected2}\n")


#        for i in range(0, len(self.symbols)):
#            if (detected[i] != detected2[i]):
#                print(f"シンボルの検出結果が違います。detected={detected[i]}, detected2={detected2[i]}\n")

        debugImgWrite(marks, "step5", "marks3_detected")
        
        return(detected)     
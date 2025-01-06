import cv2
import numpy as np
import math
import json
import glob
import csv
import os
import sys
import inspect
from typing import List, Tuple
from dataclasses import dataclass, field, asdict
from typing import ClassVar
from dataclasses_json import dataclass_json, config
from marshmallow import Schema, fields

import tensorflow as tf
from tensorflow.keras.models import load_model


from ..util import *
# import scanreport.util.utility as util



# マークの状態を示すenum
from enum import Enum
class MarkStatus(Enum):
    NO = 0
    UNCERTUN = 1
    YES = 2

    def symbol(self) -> str:
        if (self == MarkStatus.YES):
            return('O')
        elif (self == MarkStatus.UNCERTUN):
            return('?')
        else:
            return(' ')
        
    def __str__(self) -> str:
        return self.symbol()
    
    def to_int(self) -> int:
        return self.value
    
    @classmethod
    def read(cls, symbol: str) -> 'MarkStatus':
        if (symbol == 'O'):
            return cls.YES
        elif ((symbol == '?') or (symbol == 'E')):
            return cls.UNCERTUN
        else:
            return cls.NO
    
    @classmethod
    def int_to_status(cls, status: int) -> 'MarkStatus':
        if (status == 2):
            return cls.YES
        elif (status == 1):
            return cls.UNCERTUN
        else:
            return cls.NO




def trimImage(img:np.ndarray, trim:int) -> np.ndarray:
    """ 
    画像の端をトリムした画像を返す
    """
    return img[trim:img.shape[0]-trim, trim:img.shape[1]-trim]

def resizeImageWithMaxPooling(img:np.ndarray, width:int, height:int) -> np.ndarray:
    """
    白黒の画像を指定したサイズにリサイズする。リサイズする際には、Max Poolingのアルゴリズムを使う。
    """
    poolX = calcPoolSize(img.shape[1], width)
    poolY = calcPoolSize(img.shape[0], height)
    newImg = np.zeros((height, width), np.uint8)
    xPos = 0
    for x in range(0, width):
        yPos = 0
        for y in range(0, height):
            newImg[y, x] = np.max(img[yPos:yPos+poolY[y], xPos:xPos+poolX[x]])
            yPos += poolY[y]
        xPos += poolX[x]
    return newImg

def calcPoolSize(srcSize:int, dstSize:int) -> List[int]:
    """
    srcSizeをdstSizeに縮小するための、プーリングサイズの配列を計算する。余りが出る場合、両端に余りを振り分ける。
    """
    poolSize = []
    base_size = srcSize // dstSize
    mod_size = srcSize % dstSize
    for i in range(0, dstSize):
        poolSize.append(base_size)
        if (i < (mod_size+1)//2):
            poolSize[i] += 1
        elif (dstSize-1-i < mod_size//2):
            poolSize[i] += 1
    assert sum(poolSize) == srcSize, f"プーリングサイズの合計が元のサイズと一致しません。srcSize={srcSize}, dstSize={dstSize}, poolSize={poolSize}"
    return poolSize


# マーク画像を解析するパーサー
class MarkImagePaser2():
    ImageWidth = 32     # 正規化したマーク画像の幅
    ImageHeight = 32    # 正規化したマーク画像の高さ

    def __init__(self):
        self.headerImage = None     # ヘッダー部分の画像
        self.maskImage = None       # ヘッダー部分を元にした、マスク用の画像
        self.markImage = None       # マーク画像
        self.maskedMarkImage = None # マスク画像で処理したマーク画像
        self.model = None           # 機械学習モデル
    
    def readMarkBase(self, base):
        # マスク画像を作成する
        debugImgWrite(base, inspect.currentframe().f_code.co_name, "maskRaw")
        self.headerImage = resizeImageWithMaxPooling(base, MarkImagePaser2.ImageWidth, MarkImagePaser2.ImageHeight)
        self.maskImage = cv2.bitwise_not(self.headerImage)          # マスクにするので白黒反転する
        debugImgWrite(self.maskImage, inspect.currentframe().f_code.co_name, "maskNormarized")

    def readMark(self, img) -> MarkStatus:
        """
        マーク画像を元に、マークの状態を返す
            img: マーク画像
            return: マークの状態 (YES, UNCERTUN, NO)
        """
        debugImgWrite(img, inspect.currentframe().f_code.co_name, "markRaw")

        # マーク画像を正規化する
        self.markImage = resizeImageWithMaxPooling(img, MarkImagePaser2.ImageWidth, MarkImagePaser2.ImageHeight)
        debugImgWrite(self.markImage, inspect.currentframe().f_code.co_name, "markNormarized")

        # マーク画像をマスク画像で処理して、中央の文字の部分を消した画像を取得する
        self.maskedMarkImage = cv2.bitwise_and(self.markImage, self.maskImage)
        debugImgWrite(self.maskedMarkImage, inspect.currentframe().f_code.co_name, "markMasked")

        # マーク画像を元に、マークの状態を取得する
        status = self.parseMarkImg(self.maskedMarkImage)

        # マーク画像を機械学習モデルで解析する
        status2 = self.parseMarkImg2(self.markImage)

        if (status != status2):
            print(f"マークの状態が違います。status={status}, status2={status2}")

        return status2
    
    def parseMarkImg(self, img) -> MarkStatus:
        # マーク画像を読み取って、マークの状態を返す
        # 周辺領域の描写の割合で判断する。中央付近はシンボルをマスクしても、痕跡が残っているため、描写をカウントしない。
        
        # 周辺1/4の領域の平均値を得る
        w = img.shape[1]
        h = img.shape[0]
        ratio = 0.25
        sumTotal = np.sum(img)/255
        sumCenter = np.sum(img[int(h*ratio):int(h*(1-ratio)), int(w*ratio):int(w*(1-ratio))])/255
        sumRound = sumTotal - sumCenter
        areaRound = w*h*(1-(1-2*ratio)*(1-2*ratio))
        aveRound = sumRound/areaRound

        # マークの判定
        if (aveRound > 0.1):
            status = MarkStatus.YES
        elif (aveRound > 0.02):
            status = MarkStatus.UNCERTUN
        else:
            status = MarkStatus.NO

        # print(f"aveRound={aveRound} -> status={status.symbol()}")

        return status

    # マーク画像を機械学習モデルで解析する
    def parseMarkImg2(self, img:np.ndarray) -> MarkStatus:
        if (self.model == None):
            self.model = tf.keras.models.load_model('my_model.keras')
        assert img.shape == (32, 32), "Image shape must be (32, 32)"
        img = img.reshape(1, 32, 32, 1)
        pred = self.model.predict(img, verbose=0)
        stat = np.argmax(pred)
        if (pred[0][stat] >= 0.8):
            return MarkStatus.int_to_status(stat)
        else:
            print(f"pred = {pred}")
            return MarkStatus.UNCERTUN


class MarkImagePaerserInfo():
    def __init__(self, position:Tuple[int, int], parser:MarkImagePaser2) -> None:
        self.position = position
        self.parser = parser


class MarkListImageParser():
    """
    複数の丸印を含む画像を解析するクラス
    """
    TrimWidth = 3       # 元の画像の端をトリムする幅
    WidthRatio = 1.5    # マークの検出エリアの幅をマークの高さの何倍にするか

    def __init__(self) -> None:
        self.markPaserList = []    # マーク画像のパーサーのリスト
        self.headerListImage = None         # ヘッダ部分の画像
        self.headerMaskedListImage = None   # ヘッダ部分のマスク用の画像
        self.markListImage = None           # マーク部分の画像
        self.markMaskedListImage = None     # マーク部分をマスクした画像
    
    def readMarkHeaderImage(self, img:np.ndarray, verify_num:int=0):
        """
        マーク画像のヘッダー部分を読み取る
        """
        # 画像解析の前に画像の端をトリムする
        img = trimImage(img, MarkListImageParser.TrimWidth)
        debug_img = img.copy()
        debugImgWrite(debug_img, inspect.currentframe().f_code.co_name, "headerRaw")

        # 文字と思われる塊を検出する "蕾  花  実  胞" なら4つの塊
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        symbols = []
        detected = []
        maskWH = min(img.shape[0], img.shape[1])    # マークを検出する正方形のエリアの大きさ
        minWH = maskWH/4                            # マーク検出エリアの1/4以上の領域だけをマークの下地とみなす
        for j, contour in enumerate(contours):
            size = cv2.contourArea(contour)
            if (size > minWH * minWH):
                x,y,w,h = cv2.boundingRect(contour)
                if ((w > minWH) and (h > minWH)):
                    symbols.append((x,y,w,h))
                    detected.append(False)
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), 255, 2)   # デバッグ用に検出したシンボルを囲む
      
        # 左から順にソートする
        symbols = sorted(symbols, key=lambda s: s[0])
        debugImgWrite(debug_img, inspect.currentframe().f_code.co_name, "headerDetected")
        self.maskImageList = symbols

        # マスクの数が期待と一致するか確認する
        if (verify_num > 0):
            assert len(symbols) == verify_num, f"マスクの数({len(symbols)})が期待({verify_num})と一致しません"
        
        # 〇を検出する領域を計算する
        self.headerListImage = np.zeros((MarkImagePaser2.ImageHeight, MarkImagePaser2.ImageWidth*len(self.maskImageList)), np.uint8)
        self.headerMaskedListImage = np.zeros((MarkImagePaser2.ImageHeight, MarkImagePaser2.ImageWidth*len(self.maskImageList)), np.uint8)
        for i, symbol in enumerate(symbols):
            centerX = int(symbol[0] + symbol[2]/2)
            centerY = int(symbol[1] + symbol[3]/2)
            symbolArea = (int(max(centerX - maskWH/2 * MarkListImageParser.WidthRatio, 0)), int(max(centerY - maskWH/2, 0)), int(maskWH * MarkListImageParser.WidthRatio), maskWH)
            markParser = MarkImagePaser2()
            markParser.readMarkBase(img[symbolArea[1]:symbolArea[1]+symbolArea[3], symbolArea[0]:symbolArea[0]+symbolArea[2]])
            self.markPaserList.append(MarkImagePaerserInfo(symbolArea, markParser))
            self.headerListImage[0:MarkImagePaser2.ImageHeight, i*MarkImagePaser2.ImageWidth:(i+1)*MarkImagePaser2.ImageWidth] =  markParser.headerImage
            self.headerMaskedListImage[0:MarkImagePaser2.ImageHeight, i*MarkImagePaser2.ImageWidth:(i+1)*MarkImagePaser2.ImageWidth] =  markParser.maskImage

        debugImgWrite(self.headerListImage, inspect.currentframe().f_code.co_name, "headerList")
        debugImgWrite(self.headerMaskedListImage, inspect.currentframe().f_code.co_name, "headerListMasked")
        return self.markPaserList


    def readMarkListImage(self, img:np.ndarray) -> List[MarkStatus]:
        """
        マーク画像のリストを読み取る
        """
        # 画像解析の前に画像の端をトリムする
        img = trimImage(img, MarkListImageParser.TrimWidth)
        debug_img = img.copy()
        debugImgWrite(debug_img, inspect.currentframe().f_code.co_name, "markListRaw")
        self.markListImage = np.zeros((MarkImagePaser2.ImageHeight, MarkImagePaser2.ImageWidth*len(self.markPaserList)), np.uint8)
        self.markMaskedListImage = np.zeros((MarkImagePaser2.ImageHeight, MarkImagePaser2.ImageWidth*len(self.markPaserList)), np.uint8)

        # マーク画像を解析する
        markStatusList = []
        for i, markParserInfo in enumerate(self.markPaserList):
            markImg = img[markParserInfo.position[1]:markParserInfo.position[1]+markParserInfo.position[3], markParserInfo.position[0]:markParserInfo.position[0]+markParserInfo.position[2]]
            markStatus = markParserInfo.parser.readMark(markImg)
            self.markListImage[0:MarkImagePaser2.ImageHeight, i*MarkImagePaser2.ImageWidth:(i+1)*MarkImagePaser2.ImageWidth] =  markParserInfo.parser.markImage
            self.markMaskedListImage[0:MarkImagePaser2.ImageHeight, i*MarkImagePaser2.ImageWidth:(i+1)*MarkImagePaser2.ImageWidth] =  markParserInfo.parser.maskedMarkImage
            markStatusList.append(markStatus)
            cv2.rectangle(debug_img, (markParserInfo.position[0], markParserInfo.position[1]), (markParserInfo.position[0]+markParserInfo.position[2], markParserInfo.position[1]+markParserInfo.position[3]), 255, 2)   # デバッグ用に検出したシンボルを囲む

        debugImgWrite(debug_img, inspect.currentframe().f_code.co_name, "markListDetected")
        debugImgWrite(self.markListImage, inspect.currentframe().f_code.co_name, "markList")
        debugImgWrite(self.markMaskedListImage, inspect.currentframe().f_code.co_name, "markListMasked")
        return markStatusList

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
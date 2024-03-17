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

from utility import *
from trim_report_frame import *
import text_scan


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
        util.debugImgWrite(mask0, "step5", "mask2_mask0")
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
        util.debugImgWrite(marks, "step5", "marks0_raw")
        marks2 = cv2.blur(marks,(5,5))
        util.debugImgWrite(marks2, "step5", "marks1_blur")

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

        util.debugImgWrite(marks, "step5", "marks3_detected")
        
        return(detected)        


class TableImageParser:
    def __init__(self) -> None:
        self.cols = []
        self.rows = []
        self.src_img = []

    def parseTableImage(self, img):
        """
        表イメージを解析する
        @param img:画像イメージ(カラー)
        """
        self.src_img = img
        self.img_debug = img.copy()
        self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 画像処理のノイズ除去
        ret, self.img_gray2 = cv2.threshold(self.img_gray, 130, 255, cv2.THRESH_BINARY_INV)
        img_w = self.img_gray2.shape[1]
        img_h = self.img_gray2.shape[0]
        cv2.imwrite("./tmp/step4_gray.jpg", self.img_gray2)

        # 罫線検出
        lines = cv2.HoughLinesP(self.img_gray2, rho=1, theta=np.pi/2, threshold=80, minLineLength=img_w/4, maxLineGap=10)
        cols = []
        rows = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if ((abs(x1-x2) < 10) and (abs(y1-y2) > 10)):
                # 縦線
                cols.append(x1)
                cv2.line(self.img_debug, (x1,y1), (x2,y2), (255,0,0), 3)
            elif ((abs(x1-x2) > 10) and (abs(y1-y2) < 10)):
                # 横線
                rows.append(y1)
                cv2.line(self.img_debug, (x1,y1), (x2,y2), (0,255,0), 3)
            else:
                print(f"処理できない線がみつかりました ({x1}, {y1}-({x2},{y2}))")
                cv2.line(self.img_debug, (x1,y1), (x2,y2), (0,0,255), 3)

        # 列と行を求める
        self.cols = util.trimPosList(sorted(cols), skip=2)
        self.rows = util.trimPosList(sorted(rows), skip=2)
        util.debugImgWrite(self.img_debug, "step4", "table")

        return(self.cols, self.rows)
    
    def getCellImage(self, col:int, row:int, raw=False):
        img = self.src_img if raw else self.img_gray2
        cell_img = img[self.rows[row][1]:self.rows[row+1][0], self.cols[col][1]:self.cols[col+1][0]]
        return(cell_img)

    def getColHeaderImage(self, col:int, raw=False):
        img = self.src_img if raw else self.img_gray2
        cell_img = img[self.rows[0][1]:self.rows[1][0], self.cols[col][1]:self.cols[col+1][0]]
        return(cell_img)

    def getColContens(self, col:int, raw=False):
        img = self.src_img if raw else self.img_gray2
        cell_img = img[self.rows[1][1]:self.rows[-1][0], self.cols[col][1]:self.cols[col+1][0]]
        return(cell_img)
    
class YasouRecordTable:
    indexNo = 0
    indexName = 1
    indexStatus = 2
    indexSampling = 3
    indexNote = 4
    expectedCols = 5

    def __init__(self) -> None:
        self.img = None
        self.table_parser = None
    
    def parseImg(self, img):
        self.img = img
        self.table_parser = TableImageParser()
        self.table_parser.parseTableImage(img)
        assert len(self.table_parser.cols)-1 == YasouRecordTable.expectedCols, f"表のカラム数{len(self.table_parser.cols)-1}が期待した値{YasouRecordTable.expectedCols}と違います"


def parseMainImg(img):
    # 野草の表イメージを読み込む
    table = YasouRecordTable()
    table.parseImg(img)

    # 種名の一覧リスト
    plants_img = table.table_parser.getColContens(YasouRecordTable.indexName, raw=True)
    plant_list = []

    # 蕾花実の一覧リスト
    stats_img = table.table_parser.getColContens(YasouRecordTable.indexStatus, raw=True)
    stat_base = table.table_parser.getColHeaderImage(YasouRecordTable.indexStatus)
    stat_parser = MarkImageParser()
    stat_parser.readMarkBase(stat_base, verify_num=4)
    stat_list = []
    for i in range(1, len(table.table_parser.rows)-1, 1):
        stat = table.table_parser.getCellImage(YasouRecordTable.indexStatus, i)
        
        # 蕾花実から〇を検出する
        detected_stat = stat_parser.detectMarks(stat)
        #print(f"stat[{i}]: {str(detected_stat)}")
        stat_list.append(detected_stat)
        
    # 採取の一覧リスト
    samples_img = table.table_parser.getColContens(YasouRecordTable.indexSampling, raw=True)
    sample_base = table.table_parser.getColHeaderImage(YasouRecordTable.indexSampling)
    sample_parser = MarkImageParser()
    sample_parser.readMarkBase(sample_base, verify_num=1)
    sample_list = []
    for i in range(1, len(table.table_parser.rows)-1, 1):
        sample = table.table_parser.getCellImage(YasouRecordTable.indexSampling, i)
        
        # 〇を検出する
        detected_sample = sample_parser.detectMarks(sample)
        #print(f"sample[{i}]: {str(detected_sample)}")
        sample_list.append(detected_sample)

    # 備考の一覧リスト
    notes_img = table.table_parser.getColContens(YasouRecordTable.indexNote, raw=True)
    note_list = []


    return ((plants_img, stats_img, samples_img, notes_img), (plant_list, stat_list, sample_list, note_list))

def YesNoMark(flag):
    if (flag == 2):
        return('〇')
    elif (flag == 1):
        return('?')
    else:
        return('')


# main
g_skipText=False
if __name__ == '__main__':
    args = sys.argv
    if 2 > len(args):
        print(f"Usage {args[0]} [--skipText] image_file")
    else:
        debugTmpImgRemove()
        for i in range(1, len(args)):
            arg = args[i]
            if (arg.startswith('--')):
                if (arg == "--skipText"):
                    g_skipText = True
                else:
                    print(f"Ignored invalid option: {arg}")
            else:
                files = glob.glob(arg)
                for file in files:
                    print(f"読み込み処理開始:{file}")
                    
                    # 記録用紙内の四隅で囲まれた内容を取得する。台形補正もこの段階で実施される
                    img = cv2.imread(file)
                    trim_img = trim_paper_frame(img)
                    trim_img2 = trim_inner_mark2(trim_img)

                    # 各記録の部分を切り出す
                    main_img, head_img, place_img = getDescArea(trim_img2)
                    debugImgWrite(main_img, "step0", "main")
                    debugImgWrite(head_img, "step0", "head")
                    debugImgWrite(place_img, "step0", "place")

                    # メイン部の内容を読み取る
                    ((plant_img, stat_img, sample_img, note_img), (plants, stats, samples, notes)) = parseMainImg(main_img)
                    debugImgWrite(plant_img, "step1", "plant")
                    debugImgWrite(stat_img, "step1", "stat")
                    debugImgWrite(sample_img, "step1", "sample")
                    debugImgWrite(note_img, "step1", "note")

                    # 表の行数を適切に読み込めているかを確認する
                    assert len(stats) == 30, f"表を正しく読み取れていません: {len(stats)}"

                    # 種名の画像ファイルから種名リストを得る
                    if (not g_skipText):
                        plantnames = text_scan.ReadPlantsFromImage(plant_img, len(stats))
                    else:
                        plantnames = ["skiped plant names"] * len(stats)
                        print(f"Option: 種名の画像認識をスキップします")
                    print(f"PlantName:{plantnames}")

                    # 蕾、花、実のリストを出力する
                    csvfile = file.upper().replace(".JPG", "") + "_result.csv"
                    print(f"csvに出力 {csvfile}")
                    with open(csvfile, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([file])
                        writer.writerow(["No","区間","種名","","蕾","花","実","胞子","採種"])
                        for i, stat in enumerate(stats):
                            assert len(stat) == 4, f"蕾, 花, 実, 胞子の読み込みに失敗しています: {str(stat)}"
                            assert len(samples[i]) == 1, f"採種の読み込みに失敗しています: {str(samples[i])}"
                            writer.writerow([i+1,"",plantnames[i],"",YesNoMark(stat[0]),YesNoMark(stat[1]),YesNoMark(stat[2]),YesNoMark(stat[3]),YesNoMark(samples[i][0])])
                    
                    print(f"読み込み処理終了:{file}")

                    


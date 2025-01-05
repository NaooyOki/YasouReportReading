import cv2
import numpy as np
import math
import json
import glob
import csv
import os
import sys
import inspect
import ast
from dataclasses import dataclass, field, asdict
from typing import ClassVar, List
from dataclasses_json import dataclass_json, config
from marshmallow import Schema, fields
from .mark_parser import *
from typing import List

#from utility import *
#from trim_report_frame import *
#from text_scan import *
#from frame_info import *
#from mark_parser import *


@dataclass
class StatRecord:
    #STAT_NO: ClassVar[int] = 0
    #STAT_YES: ClassVar[int] = 2
    #STAT_UNCERTURN: ClassVar[int] = 1

    index:int = 0
    stat_tubomi:MarkStatus = MarkStatus.NO
    stat_flower:MarkStatus = MarkStatus.NO
    stat_seed:MarkStatus = MarkStatus.NO
    stat_houshi:MarkStatus = MarkStatus.NO
    sample:MarkStatus = MarkStatus.NO
    stat_image:np.ndarray = field(init=False, repr=False)
    sample_image:np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.stat_image = np.zeros((2, 2))
        self.sample_image = np.zeros((2, 2))


@dataclass
class StatReportInfo:
    file_name:str = ""
    header:StatRecord = field(init=False)
    records:List[StatRecord] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.header = StatRecord(0, MarkStatus.NO, MarkStatus.NO, MarkStatus.NO, MarkStatus.NO, MarkStatus.NO)
        self.records = []


    def write_file(self, output_path, filename):
        """
        機械学習の教師データにするために、解析した情報を画像ファイルとデータファイルの組として出力する
        output_path: 出力先のディレクトリ
        filename: 出力ファイルの名前   Stat_<filename>.csv, Stat_<filename>.jpg というファイル名で出力される
        """
        index_size = (32, 32)
        target_size = (32*5, 32)
        csvfile = os.path.join(output_path, f"Stat_{filename}.csv")
        imgfile = os.path.join(output_path, f"Stat_{filename}.jpg")

        resized_imgs = []
        print(f"csvに出力 {filename}")
        with open(csvfile, 'w', newline='', encoding='utf-8') as f:
            # write metadata
            writer = csv.writer(f)
            writer.writerow(["Stat_" + filename, "stat", index_size, target_size])
            writer.writerow(["No","T","F","M","H","S"])

            # write header
            writer.writerow([f"00", MarkStatus.NO.symbol(), MarkStatus.NO.symbol(), MarkStatus.NO.symbol(), MarkStatus.NO.symbol(), MarkStatus.NO.symbol()])
            index_img = create_numbered_image(0, index_size)
            resized_imgs.append(cv2.hconcat([index_img, self.header.stat_image, self.header.sample_image]))

            # write records
            for record in self.records:
                try:
                    writer.writerow([f"{record.index:02}", record.stat_tubomi.symbol(), record.stat_flower.symbol(), record.stat_seed.symbol(), record.stat_houshi.symbol(), record.sample.symbol()])
                    index_img = create_numbered_image(record.index, index_size)
                    resized_imgs.append(cv2.hconcat([index_img, record.stat_image, record.sample_image]))
                except Exception as e:
                    print(f"書き込みエラーが発生しました: {e}")
        
        stat_img = cv2.vconcat(resized_imgs)
        cv2.imwrite(imgfile, stat_img)



def str_to_tuple(string):
  """文字列をタプルに変換する関数"""
  try:
    return ast.literal_eval(string)
  except ValueError:
    print("不正な形式の文字列です。")
    return None

def read_statreport_file(dir, filename):
    report_info = StatReportInfo(filename)
    csvfile = os.path.join(dir, f"{filename}.csv")
    imgfile = os.path.join(dir, f"{filename}.jpg")

    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    
    print(f"ファイルを読み込み  {filename}")
    with open(csvfile, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        index_size = (0,0)
        target_size = (0,0)
        for i, row in enumerate(reader):
            if (i == 0):
                # メタ情報
                index_size = str_to_tuple(row[2])
                target_size = str_to_tuple(row[3])
                #print(f"index_size={index_size}, target_size={target_size}")
            elif (i > 1):
                # レコード情報
                no = int(row[0])
                stat_t = MarkStatus.read(row[1])
                stat_f = MarkStatus.read(row[2])
                stat_m = MarkStatus.read(row[3])
                stat_h = MarkStatus.read(row[4])
                samp = MarkStatus.read(row[5])
                record = StatRecord(no, stat_t, stat_f, stat_m, stat_h, samp)
                cropped_img = img[target_size[1]*no:target_size[1]*(no+1), index_size[0]:index_size[0]+target_size[0]]
                record.stat_image = cropped_img
                report_info.records.append(record)

                #print(f"record={record}")
                #cv2.imshow('Cropped Image', cropped_img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

    return report_info


                




def create_numbered_image(no, img_size):
    """
    黒背景に行番号を記述した画像を作成する関数
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 2

    # 行番号を画像の中央に配置するために、フォントの起点を計算する
    text_size = cv2.getTextSize(str(no), font, font_scale, font_thickness)[0]
    text_x = int((img_size[0] - text_size[0]) / 2)
    text_y = int((img_size[1] + text_size[1]) / 2)

    # 行番号を中央に記述した画像を作成する
    img = np.zeros(img_size, dtype=np.uint8)  # 黒背景の画像を作成
    cv2.putText(img, str(no), (text_x, text_y), font, font_scale, 255, font_thickness)
    return img    
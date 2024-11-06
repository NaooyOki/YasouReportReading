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

#from utility import *
#from trim_report_frame import *
#from text_scan import *
#from frame_info import *
#from mark_parser import *


@dataclass
class StatRecord:
    STAT_NO: ClassVar[int] = 0
    STAT_YES: ClassVar[int] = 2
    STAT_UNCERTURN: ClassVar[int] = 1

    index:int = 0
    stat:List[int] = field(default_factory=list)
    stat_tubomi:int = field(default=STAT_NO, init=False)
    stat_flower:int = field(default=STAT_NO, init=False)
    stat_seed:int = field(default=STAT_NO, init=False)
    stat_houshi:int = field(default=STAT_NO, init=False)
    sample:int = STAT_NO
    stat_image:np.ndarray = field(init=False, repr=False)
    sample_image:np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        assert len(self.stat) == 4
        self.stat_tubomi = self.stat[0]
        self.stat_flower = self.stat[1]
        self.stat_seed = self.stat[2]
        self.stat_houshi = self.stat[3]
        self.stat_image = np.zeros((2, 2))
        self.sample_image = np.zeros((2, 2))


    def YesNoMark(flag:int) -> str:
        if (flag == 2):
            return('O')
        elif (flag == 1):
            return('?')
        else:
            return(' ')
        
    def MarkToFlag(mark:str) -> int:
        if (mark == "O"):
            return(StatRecord.STAT_YES)
        elif (mark == ' '):
            return(StatRecord.STAT_NO)
        else:
            return(StatRecord.STAT_UNCERTURN)



@dataclass
class StatReportInfo:
    file_name:str = ""
    header:StatRecord = field(init=False)
    records:List[StatRecord] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.header = StatRecord(0, [0, 0, 0, 0], 0)
        self.records = []


    def write_file(self, output_path, filename):
        index_size = (64, 64)
        target_size = (370, 64)
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
            writer.writerow([f"00", StatRecord.YesNoMark(0),StatRecord.YesNoMark(0),StatRecord.YesNoMark(0),StatRecord.YesNoMark(0),StatRecord.YesNoMark(0)])
            resize_img = cv2.resize(self.header.stat_image, target_size)
            index_img = create_numbered_image(0, index_size)
            resized_imgs.append(cv2.hconcat([index_img, resize_img]))

            # write records
            for record in self.records:
                try:
                    writer.writerow([f"{record.index:02}", StatRecord.YesNoMark(record.stat_tubomi),StatRecord.YesNoMark(record.stat_flower),StatRecord.YesNoMark(record.stat_seed),StatRecord.YesNoMark(record.stat_houshi),StatRecord.YesNoMark(record.sample)])
                    resize_img = cv2.resize(record.stat_image, target_size)
                    index_img = create_numbered_image(record.index, index_size)
                    resized_imgs.append(cv2.hconcat([index_img, resize_img]))
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

    img = cv2.imread(imgfile)
    
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
                stat_t = StatRecord.MarkToFlag(row[1])
                stat_f = StatRecord.MarkToFlag(row[2])
                stat_m = StatRecord.MarkToFlag(row[3])
                stat_h = StatRecord.MarkToFlag(row[4])
                samp = StatRecord.MarkToFlag(row[5])
                record = StatRecord(no, [stat_t, stat_f, stat_m, stat_h], samp)
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
    img = np.zeros(img_size, dtype=np.uint8)  # 黒背景の画像を作成
    cv2.putText(img, str(no), (10, int(img_size[1]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    return img    
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
from text_scan import *
from frame_info import *
from mark_parser import *


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
    stat_image:np.ndarray = field(default_factory=np.zeros)
    sample_image:np.ndarray = field(default_factory=np.zeros)

    def __post_init__(self):
        assert len(self.stat) == 4
        self.stat_tubomi = self.stat[0]
        self.stat_flower = self.stat[1]
        self.stat_seed = self.stat[2]
        self.stat_houshi = self.stat[3]

    def YesNoMark(flag:int) -> str:
        if (flag == 2):
            return('O')
        elif (flag == 1):
            return('?')
        else:
            return(' ')



@dataclass
class StatReportInfo:
    file_name:str = ""
    header:StatRecord = field(init=False)
    records:List[StatRecord] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.header = StatRecord(0, [0, 0, 0, 0], 0, np.zeros((2,2)), np.zeros((2,2)))
        self.records = []



                    

def scan_report(target_file:str) -> StatReportInfo:
    report_info = StatReportInfo(file_name=target_file)

    img = cv2.imread(target_file)
    trim_img = trim_report_frame.trim_paper_frame(img)
    trim_img2, found = trim_report_frame.trim_inner_mark2(trim_img)
    if (not found):
        trim_img2 = trim_report_frame.trim_inner_mark(trim_img)

    # 画像をグレースケールにして白黒反転する
    img_gray = cv2.cvtColor(trim_img2, cv2.COLOR_BGR2GRAY)
    ret, scan_image = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # 画像を読み取り、フレーム情報を読み取る
    frame_detector = FrameDetector()
    root = frame_detector.parse_image(trim_img2)   
    root_child = frame_detector.detect_sub_frames(root, 0)
    #print(f"root_child={root_child}")

    # メイン部分を取り出す
    main = root.get_frame_in_cluster(2, 0)
    main_child = frame_detector.detect_table_frame(main, 1)
    #main_child = frame_detector.detect_sub_frames(main, 1)
    #print(f"main_child1={main_child}")

    # 蕾花実列用のパーサーを作成
    main_stat_header:Frame = main.get_frame_in_cluster(0, 2)
    stats_header_image = main_stat_header.get_image(scan_image)
    stat_parser = MarkImageParser()
    stat_parser.readMarkBase(stats_header_image, verify_num=4)
    report_info.header.stat_image = stats_header_image

    # 採取列用のパーサーを作成
    main_samp_header:Frame = main.get_frame_in_cluster(0, 3)
    samp_header_image = main_samp_header.get_image(scan_image)
    samp_parser = MarkImageParser()
    samp_parser.readMarkBase(samp_header_image, verify_num=1)
    report_info.header.sample_image = samp_header_image

    # メイン部分
    for row_index in range(1, len(main.cluster_list_row)):
        #main_no = main.get_frame_in_cluster(row_index, 0)
        #main_plant = main.get_frame_in_cluster(row_index, 1)
        
        main_stat = main.get_frame_in_cluster(row_index, 2)
        stat_image = main_stat.get_image(scan_image)
        detected_stat = stat_parser.detectMarks(stat_image)
        main_stat.value = detected_stat

        main_sample = main.get_frame_in_cluster(row_index, 3)
        samp_image = main_sample.get_image(scan_image)
        detected_samp = samp_parser.detectMarks(samp_image)
        main_sample.value = detected_samp

        main_note = main.get_frame_in_cluster(row_index, 4)
        try:
            assert len(detected_stat) == 4, f"蕾, 花, 実, 胞子の読み込みに失敗しています: {str(detected_stat)}"
            assert len(detected_samp) == 1, f"採種の読み込みに失敗しています: {str(detected_samp)}"
            # print(f"No:{row_index}, stat_img:{stat_image.shape}, stat:{main_stat.value}, sample:{main_sample.value}")
            
            record = StatRecord(index=row_index, stat=detected_stat, sample=detected_samp[0], stat_image=stat_image, sample_image=samp_image)
            record.stat=detected_stat
            record.stat_image = stat_image
            record.sample = detected_samp[0]
            record.sample_image = samp_image
            report_info.records.append(record)
        except:
            print(f"skip index={row_index}")
 
    return report_info  


def output_stat_info(report_info: StatReportInfo, filename:str):
    index_size = (64, 64)
    target_size = (370, 64)
    output_path = "./stat_img"
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
        resize_img = cv2.resize(report_info.header.stat_image, target_size)
        index_img = create_numbered_image(0, index_size)
        resized_imgs.append(cv2.hconcat([index_img, resize_img]))

        # write records
        for record in report_info.records:
            try:
                writer.writerow([f"{record.index:02}", StatRecord.YesNoMark(record.stat_tubomi),StatRecord.YesNoMark(record.stat_flower),StatRecord.YesNoMark(record.stat_seed),StatRecord.YesNoMark(record.stat_houshi),StatRecord.YesNoMark(record.sample)])
                resize_img = cv2.resize(record.stat_image, target_size)
                index_img = create_numbered_image(record.index, index_size)
                resized_imgs.append(cv2.hconcat([index_img, resize_img]))
            except Exception as e:
                print(f"書き込みエラーが発生しました: {e}")
    
    stat_img = cv2.vconcat(resized_imgs)
    cv2.imwrite(imgfile, stat_img)

def create_numbered_image(no, img_size):
    """
    黒背景に行番号を記述した画像を作成する関数
    Args:
        no: 列数
        img_width: 画像の幅
        img_height: 画像の高さ

    Returns:
        np.ndarray: 生成された画像
    """
    img = np.zeros(img_size, dtype=np.uint8)  # 黒背景の画像を作成
    cv2.putText(img, str(no), (10, int(img_size[1]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    return img    


# main
g_skipText = False
if __name__ == '__main__':
    # 引数の読み取り処理
    args = sys.argv
    if 3 < len(args):
        print(f"Usage {args[0]} image_file")
        exit(-1)
    for i in range(1, len(args)):
        arg = args[i]
        if (arg.startswith('--')):
            if (arg == "--skipText"):
                g_skipText = True
            else:
                print(f"Ignored invalid option: {arg}")
        else:
            files = glob.glob(arg)

    debugTmpImgRemove()
    #files = ["./record/202403/202403B01.JPG", "./record/202403/202403B02.JPG"]

    for file in files:
        print(f"読み込み処理開始:{file}")
        report = scan_report(file)
        print(f"読み込み処理終了:{file}")
        filename, ext = os.path.splitext(os.path.basename(file))
        output_stat_info(report, filename)


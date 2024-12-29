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


from .status_train_info import *
from ..util import *
from ..frame.trim_report_frame import *
from ..frame.frame_info import *
from ..frame import *
from .mark_parser import *

#scanreport.frame.trim_report_frame import *
#from scanreport.frame.text_scan import *
#from scanreport.frame.frame_info import *
#from scanreport.mark.mark_parser import *



                    

def create_traindata_from_report(target_file:str) -> StatReportInfo:
    report_info = StatReportInfo(file_name=target_file)

    img = cv2.imread(target_file)
    trim_img = trim_paper_frame(img)
    trim_img2, found = trim_inner_mark2(trim_img)
    if (not found):
        trim_img2 = trim_inner_mark(trim_img)

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




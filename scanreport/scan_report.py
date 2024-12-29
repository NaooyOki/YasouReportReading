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
from typing import ClassVar, List, Dict
from dataclasses_json import dataclass_json, config
from marshmallow import Schema, fields

from .util import *
from .frame import *
from .mark import *


#from scanreport.util.utility import *
#from scanreport.frame.trim_report_frame import *
#from scanreport.frame.text_scan import *
#from scanreport.frame.frame_info import *
#from scanreport.mark.mark_parser import *


@dataclass
class YasouRecord:
    STAT_NO: ClassVar[int] = 0
    STAT_YES: ClassVar[int] = 2
    STAT_UNCERTURN: ClassVar[int] = 1

    index:int = 0
    plant_name:str = ""
    stat:List[int] = field(default_factory=list)
    stat_tubomi:int = field(default=STAT_NO, init=False)
    stat_flower:int = field(default=STAT_NO, init=False)
    stat_seed:int = field(default=STAT_NO, init=False)
    stat_houshi:int = field(default=STAT_NO, init=False)
    sample:int = STAT_NO
    note:str = ""

    def __post_init__(self):
        assert len(self.stat) == 4
        self.stat_tubomi = self.stat[0]
        self.stat_flower = self.stat[1]
        self.stat_seed = self.stat[2]
        self.stat_houshi = self.stat[3]

    def YesNoMark(flag:int) -> str:
        if (flag == 2):
            return('〇')
        elif (flag == 1):
            return('？')
        else:
            return('')



@dataclass
class YasouReportInfo:
    date:str
    date_year:int = field(default=0, init=False)
    date_month:int = field(default=0, init=False)
    date_day:int = field(default=0, init=False)
    weather:str = ""
    course_name:str = ""
    course_page:int = 0
    member:str = ""
    records:List[YasouRecord] = field(default_factory=list, init=False)

    def __post_init__(self):
        # 年、月、日を抽出する
        pattern = r".*(\d{4})年(\d{1,2})月(\d{1,2})日.*"
        match = re.match(pattern, self.date)
        if match:
            # 年、月、日を取得
            self.date_year = int(match.group(1))
            self.date_month = int(match.group(2))
            self.date_day = int(match.group(3))




    

                    

def scan_report(target_file:str) -> YasouReportInfo:
    cache_file = "./cache/" + os.path.basename(target_file) + ".pickle"
    img = cv2.imread(target_file)
    trim_img = trim_report_frame.trim_paper_frame(img)
    trim_img2, found = trim_report_frame.trim_inner_mark2(trim_img)
    if (not found):
        trim_img2 = trim_report_frame.trim_inner_mark(trim_img)

    # 画像をグレースケールにして白黒反転する
    img_gray = cv2.cvtColor(trim_img2, cv2.COLOR_BGR2GRAY)
    ret, scan_image = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # OCR機能を使ってテキスト情報を読み取る
    img_reader = text_scan.VisonImgTextReader()
    if (os.path.exists(cache_file)):
        print("read cache")
        img_reader.load_file(cache_file)
    else:
        print("scan and create cache")
        img_reader.read_image(trim_img2)
        img_reader.save_file(cache_file)
    
    # 画像を読み取り、フレーム情報を読み取る
    frame_detector = FrameDetector()
    root = frame_detector.parse_image(trim_img2)   
    root_child = frame_detector.detect_sub_frames(root, 0)
    #print(f"root_child={root_child}")

    # ヘッダを取り出す
    head = root.get_frame_in_cluster(0, 0)
    head_child = frame_detector.detect_sub_frames(head, 1)
    #print(f"head_child={root_child}")
    head_date = head.get_frame_in_cluster(0, 0)
    head_member = head.get_frame_in_cluster(1, 0)
    head_wed = head.get_frame_in_cluster(0, 1)

    # コースを取り出す
    course = root.get_frame_in_cluster(1, 0)
    course_child = frame_detector.detect_sub_frames(course, 1)
    #print(f"course_child={course_child}")
    course_route = course.get_frame_in_cluster(0, 0)
    course_page = course.get_frame_in_cluster(0, 1)

    # メイン部分を取り出す
    main = root.get_frame_in_cluster(2, 0)
    main_child = frame_detector.detect_table_frame(main, 1)
    #main_child = frame_detector.detect_sub_frames(main, 1)
    #print(f"main_child1={main_child}")

    # 取得したフレーム内の文字をまとめて読み込む
    scan_frame(root, img_reader)

    # 取り出した情報を表示する
     
    print(f"head_date: {safe_value(head_date)}")
    print(f"head_member: {safe_value(head_member)}")
    print(f"head_wed: {safe_value(head_wed)}")
    
    print(f"course_route: {safe_value(course_route)}")
    print(f"course_page: {safe_value(course_page)}")

    # 蕾花実列用のパーサーを作成
    main_stat_header:Frame = main.get_frame_in_cluster(0, 2)
    stats_header_image = main_stat_header.get_image(scan_image)
    stat_parser = MarkImageParser()
    stat_parser.readMarkBase(stats_header_image, verify_num=4)

    # 採取列用のパーサーを作成
    main_samp_header:Frame = main.get_frame_in_cluster(0, 3)
    samp_header_image = main_samp_header.get_image(scan_image)
    samp_parser = MarkImageParser()
    samp_parser.readMarkBase(samp_header_image, verify_num=1)

    # メイン部分
    for row_index in range(1, len(main.cluster_list_row)):
        main_no = main.get_frame_in_cluster(row_index, 0)
        main_plant = main.get_frame_in_cluster(row_index, 1)
        
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
            print(f"No:{row_index}, plant:{main_plant.value}, stat:{main_stat.value}, sample:{main_sample.value}, note:{main_note.value}")
        except:
            print(f"skip index={row_index}")

    # 読み取り結果を出力する
    route = get_match_value(course_route.value, r"([A-Z])\s*コース", "X")
    page = get_match_value(course_page.value, r"(\d+)\s*枚目", "99")
    member = get_match_value(head_member.value, r"調査者:\s*(.+)", "")
    report_info = YasouReportInfo(date=head_date.value, weather=head_wed.value, course_name=route, course_page=page, member=member)
    for row_index in range(1, len(main.cluster_list_row)):
        main_no = main.get_frame_in_cluster(row_index, 0)
        main_plant = main.get_frame_in_cluster(row_index, 1)
        main_stat = main.get_frame_in_cluster(row_index, 2)
        stat = main_stat.value
        main_samp = main.get_frame_in_cluster(row_index, 3)
        samp = main_samp.value
        main_note = main.get_frame_in_cluster(row_index, 4)
        assert len(stat) == 4, f"蕾, 花, 実, 胞子の読み込みに失敗しています: {str(stat)}"
        assert len(samp) == 1, f"採種の読み込みに失敗しています: {str(samp)}"
        record = YasouRecord(index=row_index, plant_name=main_plant.value, stat=stat, sample=samp[0], note=main_note.value)
        report_info.records.append(record)

    return report_info  


def output_csv_report(report_info: YasouReportInfo, csvfile:str):
    print(f"csvに出力 {csvfile}")
    with open(csvfile, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([report_info.date])
        writer.writerow([report_info.member])
        writer.writerow([report_info.course_name, report_info.course_page])
        writer.writerow(["No","区間","種名","","蕾","花","実","胞子","採種","備考"])
        for record in report_info.records:
            try:
                writer.writerow([record.index, report_info.course_name, record.plant_name, "", YasouRecord.YesNoMark(record.stat_tubomi),YasouRecord.YesNoMark(record.stat_flower),YasouRecord.YesNoMark(record.stat_seed),YasouRecord.YesNoMark(record.stat_houshi),YasouRecord.YesNoMark(record.sample), record.note])
            except Exception as e:
                print(f"書き込みエラーが発生しました: {e}")

# main
g_skipText = False
def main():
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
    # files = ["./record/202403/202403B01.JPG", "./record/202403/202403B02.JPG"]

    report_list:Dict[str, List[YasouReportInfo]] = dict()
    folder = os.path.dirname(files[0])

    for file in files:
        print(f"読み込み処理開始:{file}")
        report = scan_report(file)

        # コース別にレポートのリストを作成する
        if (report.course_name in report_list):
            report_list[report.course_name].append(report)
        else:
            report_list[report.course_name] = [report]

        print(f"読み込み処理終了:{file}")


    for course_reports in report_list.values():
        course_reports.sort(key=lambda report: report.course_page)
        marged_report:YasouReportInfo = course_reports[0]
        for report in course_reports[1:]:
            marged_report.records.extend(report.records)
            marged_report.course_page += f",{report.course_page}"
            if (marged_report.member == ""):
                marged_report.member = report.member

        csvfile = os.path.join(folder, f"{marged_report.date_year:04}{marged_report.date_month:02}{marged_report.course_name}.csv")

        print(f"CSVファイルに出力:{csvfile}")
        output_csv_report(marged_report, csvfile)


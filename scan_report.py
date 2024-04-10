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
        self.cols = trimPosList(sorted(cols), skip=2)
        self.rows = trimPosList(sorted(rows), skip=2)
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
    
def trimPosList(lst:List[int], skip:int = 2) -> List[int]:
    """
    隣接する直線を一つに束ねる。表の罫線は太いため、直線検出では複数の直線として得られるため。
    @param list:List[int]  線の位置
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


def parseMainImg(img, img_pos):
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
if __name__ == '__main2__':
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
                    main_img, main_pos, head_img, head_pos, place_img, place_pos = getDescArea(trim_img2)
                    debugImgWrite(main_img, "step0", "main")
                    debugImgWrite(head_img, "step0", "head")
                    debugImgWrite(place_img, "step0", "place")

                    # メイン部の内容を読み取る
                    ((plant_img, stat_img, sample_img, note_img), (plants, stats, samples, notes)) = parseMainImg(main_img, main_pos)
                    debugImgWrite(plant_img, "step1", "plant")
                    debugImgWrite(stat_img, "step1", "stat")
                    debugImgWrite(sample_img, "step1", "sample")
                    debugImgWrite(note_img, "step1", "note")

                    # 表の行数を適切に読み込めているかを確認する
                    assert len(stats) == 30, f"表を正しく読み取れていません: {len(stats)}"

                    # 種名の画像ファイルから種名リストを得る
                    if (not g_skipText):
                        cache_file = file + ".picke"
                        plantnames = text_scan.ReadPlantsFromImage(plant_img, len(stats), cache_file)
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

@dataclass
class YasouRecord:
    STAT_NO: ClassVar[int] = 0
    STAT_YES: ClassVar[int] = 2
    STAT_UNCERTURN: ClassVar[int] = 1

    index:int = 0
    plant_name:str = ""
    stat_tubomi:int = 0
    stat_flower:int = 0
    stat_seed:int = 0
    stat_houshi:int = 0
    note:str = ""

@dataclass
class YasouReportInfo:
    date:str
    date_year:str = field(default="", init=False)
    date_month:str = field(default="", init=False)
    date_day:str = field(default="", init=False)
    course_name:str
    course_page:int
    member:str
    records:List[YasouRecord] = field(default_factory=list, init=False)

    def __post_init__(self):
        # 年、月、日を抽出する
        pattern = r"(\d{4})年(\d{1,2})月(\d{1,2})日"
        match = re.match(pattern, self.date)
        if match:
            # 年、月、日を取得
            self.date_year = match.group(1)
            self.date_month = match.group(2)
            self.date_day = match.group(3)



    

                    
# main
def scan_report(target_file:str) -> YasouReportInfo:
    cache_file = "./cache/" + os.path.basename(target_file) + ".pickle"
    img = cv2.imread(target_file)
    trim_img = trim_report_frame.trim_paper_frame(img)
    trim_img2 = trim_report_frame.trim_inner_mark2(trim_img)

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
    print(f"root_child={root_child}")

    # ヘッダを取り出す
    head = root.get_frame_in_cluster(0, 0)
    head_child = frame_detector.detect_sub_frames(head, 1)
    print(f"head_child={root_child}")
    head_date = head.get_frame_in_cluster(0, 0)
    head_member = head.get_frame_in_cluster(1, 0)
    head_wed = head.get_frame_in_cluster(0, 1)

    # コースを取り出す
    course = root.get_frame_in_cluster(1, 0)
    course_child = frame_detector.detect_sub_frames(course, 1)
    print(f"course_child={course_child}")
    course_route = course.get_frame_in_cluster(0, 0)
    course_page = course.get_frame_in_cluster(0, 1)

    # メイン部分を取り出す
    main = root.get_frame_in_cluster(2, 0)
    main_child = frame_detector.detect_table_frame(main, 1)
    #main_child = frame_detector.detect_sub_frames(main, 1)
    print(f"main_child1={main_child}")

    # 取得したフレーム内の文字をまとめて読み込む
    scan_frame(root, img_reader)

    # 取り出した情報を表示する
    print(f"head_date: {head_date.value}")
    print(f"head_member: {head_member.value}")
    print(f"head_wed: {head_wed.value}")
    
    print(f"course_route: {course_route.value}")
    print(f"course_page: {course_page.value}")

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
            print(f"No:{main_no.value}, plant:{main_plant.value}, kind:{main_stat.value}, sample:{main_sample.value}, note:{main_note.value}")
        except:
            print(f"skip index={row_index}")

    # 読み取り結果を出力する
    report_info = YasouReportInfo(date=head_date.value, course_name=course_route.value, course_page=course_page.value, member=head_member.value)
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
        record = YasouRecord(index=int(main_no.value), plant_name=main_plant.value, stat_tubomi=stat[0], stat_flower=stat[1], stat_seed=stat[2], stat_houshi=stat[3], note=main_note.value)
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
            writer.writerow([record.index, record.plant_name, "",YesNoMark(record.stat_tubomi),YesNoMark(record.stat_flower),YesNoMark(record.stat_seed),YesNoMark(record.stat_houshi),YesNoMark(0), record.note])


# main
if __name__ == '__main__':
    test_file = "./record/202403/202403B01.JPG"
    report_info = scan_report(test_file)

    csvfile = test_file.upper().replace(".JPG", "") + "_result.csv"
    output_csv_report(report_info, csvfile)


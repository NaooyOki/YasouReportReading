import cv2
import numpy as np
import math
import json
import glob
import csv
import os
import sys
import inspect
import logging
import argparse

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

g_skipText = False
g_forceDate = ""
g_forceCourse = ""
g_pageCount = 0
g_allOutput = False

def setup_logger():
    # ログを設定
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # フォーマットを指定
    formatter = logging.Formatter('%(asctime)s, %(name)s, %(levelname)s, %(message)s')
    formatter.datefmt = '%Y-%m-%d %H:%M:%S'
    
    # コンソール出力の設定
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # ファイル出力の設定
    file_hander = logging.FileHandler('scanreport.log', encoding='utf-8')
    file_hander.setLevel(logging.DEBUG)
    file_hander.setFormatter(formatter)
    logger.addHandler(file_hander)


@dataclass
class YasouRecord:
    #STAT_NO: ClassVar[int] = 0
    #STAT_YES: ClassVar[int] = 2
    #STAT_UNCERTURN: ClassVar[int] = 1

    index:int = 0
    plant_name:str = ""
    stat:List[MarkStatus] = field(default_factory=list)
    stat_tubomi:MarkStatus = field(default=MarkStatus.NO, init=False)
    stat_flower:MarkStatus = field(default=MarkStatus.NO, init=False)
    stat_seed:MarkStatus = field(default=MarkStatus.NO, init=False)
    stat_houshi:MarkStatus = field(default=MarkStatus.NO, init=False)
    sample:MarkStatus = MarkStatus.NO
    note:str = ""

    def __post_init__(self):
        assert len(self.stat) == 4
        self.stat_tubomi = self.stat[0]
        self.stat_flower = self.stat[1]
        self.stat_seed = self.stat[2]
        self.stat_houshi = self.stat[3]

    def YesNoMark(flag:MarkStatus) -> str:
        return flag.symbol()

@dataclass
class YasouReportInfo:
    date_year:int = 0
    date_month:int = 0
    date_day:int = 0
    weather:str = ""
    course_name:str = ""
    course_page:int = 0
    member:str = ""
    records:List[YasouRecord] = field(default_factory=list, init=False)

def get_date_from_text(text:str) -> List[str]:
    pattern = r".*(\d{4})年(\d{1,2})月(\d{1,2})日.*"
    match = re.match(pattern, text)
    if match:
        return [match.group(1), match.group(2), match.group(3)]
    else:
        return ["", "", ""]


def trim_report_image(file:str) -> np.ndarray:
    img = cv2.imread(file)

    # 暗い背景から紙の白い部分を切り抜く
    trim_img = trim_report_frame.trim_paper_frame(img)

    # 四隅のマークで記録部分を切り出す
    trim_img2, found = trim_report_frame.trim_inner_mark2(trim_img)
    if (not found):
        trim_img2, found = trim_report_frame.trim_inner_mark(trim_img)
        if (not found):
            logger.warning(f"四隅のマークが見つかりませんでした。このファイルの解析をスキップします。: {file}")
            return None
    return trim_img2


def scan_report(img:np.ndarray, file_name:str, oldMarkParser:bool=False) -> YasouReportInfo:
    """
    野草の調査用紙を読み取り、レポートに取り込むための情報を取得する
    @param target_file:読み取る画像ファイル
    @return YasouReportInfo
    """
    global g_skipText
    global g_forceDate
    global g_forceCourse
    global g_pageCount
    global g_allOutput

    debugImgWrite(img, inspect.currentframe().f_code.co_name, "0img")

    # 通算ページ数
    g_pageCount += 1

    # 画像をグレースケールにして白黒反転する
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, scan_image = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # OCR機能を使ってテキスト情報を読み取る
    img_reader = text_scan.VisonImgTextReader()
    textCacheFile = "./cache/" + os.path.basename(file_name) + ".pickle"     # テキスト情報をキャッシュするファイル
    readCache = False
    if (os.path.exists(textCacheFile)):
        if (os.path.getmtime(textCacheFile) > os.path.getmtime(file_name)):
            readCache = True
    if (readCache):
        logger.debug(f"read cache from {textCacheFile}")
        img_reader.load_file(textCacheFile)
    else:
        logger.debug("scan and create cache")
        img_reader.read_image(img)
        img_reader.save_file(textCacheFile)

    # スキーマーを読み込む
    schema_file = "./template/PlantsInspectReport_schema.json"
    schemas = []
    with open(schema_file, "r", encoding="utf-8") as f:
        json_schemas = json.load(f)
        for s in json_schemas:
            schema = TableSchema.from_json(s)
            schemas.append(schema)
    
    # 帳票を作成する
    sheet = RecordSheet.create_sheet_from_img(scan_image, schemas)
    sheet.match_tables()

    # ヘッダを取り出す
    head = sheet.matched_tables["Header"]
    if (head is not None):
        head_date = head.get_cell(0, 0)
        head_date_text = img_reader.extract_text_from_region2(head_date.rect)
        head_member = head.get_cell(0, 1)
        head_member_text = img_reader.extract_text_from_region2(head_member.rect)
        head_wed = head.get_cell(1, 0)
        head_wed_text = img_reader.extract_text_from_region2(head_wed.rect)
    
    # コースを取り出す
    course = sheet.matched_tables["Course"]
    if (course is not None):
        course_route = course.get_cell(0, 0)
        course_route_text = img_reader.extract_text_from_region2(course_route.rect)
        course_page = course.get_cell(0, 1)
        course_page_text = img_reader.extract_text_from_region2(course_page.rect)
    

    # メイン部分を取り出す
    main = sheet.matched_tables["Main"]

    # 取り出した情報を表示する
    logger.debug(f"head_date: {head_date_text}")
    logger.debug(f"head_member: {head_member_text}")
    logger.debug(f"head_wed: {head_wed_text}")
    
    logger.debug(f"course_route: {course_route_text}")
    logger.debug(f"course_page: {course_page_text}")

    # 蕾花実列用のパーサーを作成
    main_stat_header = main.get_cell(0, 2)
    stats_header_image = sheet.get_image(main_stat_header.rect)
    debugImgWrite(stats_header_image, inspect.currentframe().f_code.co_name, "2statHeaderImg")
    stat_parser = MarkListImageParser(oldMarkParser)
    stat_parser.oldParser = oldMarkParser
    stat_parser.readMarkHeaderImage(stats_header_image, verify_num=4)

    # 採取列用のパーサーを作成
    main_samp_header = main.get_cell(0, 3)
    stats_samp_image = sheet.get_image(main_samp_header.rect)
    samp_parser = MarkListImageParser(oldMarkParser)
    samp_parser.readMarkHeaderImage(stats_samp_image, verify_num=1)

    # デバッグ用にステータス画像を記録する領域を用意する
    debug_stat_img = np.zeros((MarkImagePaser2.ImageHeight*main.row_count, MarkImagePaser2.ImageWidth*4), np.uint8)
    debug_stat_img[0:MarkImagePaser2.ImageHeight, 0:MarkImagePaser2.ImageWidth*4] = stat_parser.headerListImage
    debug_stat_masked_img = np.zeros((MarkImagePaser2.ImageHeight*main.row_count, MarkImagePaser2.ImageWidth*4), np.uint8)
    debug_stat_masked_img[0:MarkImagePaser2.ImageHeight, 0:MarkImagePaser2.ImageWidth*4] = stat_parser.headerMaskedListImage

    # 値を切り出す
    if (g_forceCourse != ""):
        route = g_forceCourse
        page = str(g_pageCount)
        logger.info(f"強制コース:{route}コース {page}枚目")
    else:
        route = get_match_value(course_route_text, r"([A-Z])\s*コース", "X")
        page = get_match_value(course_page_text, r"(\d+)\s*枚目", "99")
    
    member = get_match_value(head_member_text, r"調査者:\s*(.+)", "")
    if (g_forceDate != ""):
        logger.info(f"強制日付:{g_forceDate}")
        year = int(g_forceDate[0:4])
        month = int(g_forceDate[4:6])
        day = 1
    else:
        year, month, day = get_date_from_text(head_date_text)
    logger.info(f"日付:{year}年{month}月{day}日")

    # 読み取り結果を出力する
    report_info = YasouReportInfo(date_year=year, date_month=month, date_day=day, weather=head_wed_text, course_name=route, course_page=page, member=member)

    # メイン部分
    for row_index in range(1, main.row_count):
        main_no = main.get_cell(row_index, 0)
        main_no_text = img_reader.extract_text_from_region2(main_no.rect)

        main_plant = main.get_cell(row_index, 1)
        main_plant_text = img_reader.extract_text_from_region2(main_plant.rect)
        
        main_stat = main.get_cell(row_index, 2)
        stat_image = sheet.get_image(main_stat.rect)
        main_stat_value = stat_parser.readMarkListImage(stat_image)
      
        debug_stat_img[MarkImagePaser2.ImageHeight*(row_index):MarkImagePaser2.ImageHeight*(row_index+1), 0:MarkImagePaser2.ImageWidth*4] = stat_parser.markListImage
        debug_stat_masked_img[MarkImagePaser2.ImageHeight*(row_index):MarkImagePaser2.ImageHeight*(row_index+1), 0:MarkImagePaser2.ImageWidth*4] = stat_parser.markMaskedListImage

        main_sample = main.get_cell(row_index, 3)
        samp_image = sheet.get_image(main_sample.rect)
        main_samp_value = samp_parser.readMarkListImage(samp_image)

        main_note = main.get_cell(row_index, 4)
        main_note_text = img_reader.extract_text_from_region2(main_note.rect)

        try:
            logger.debug(f"No:{row_index}, plant:{main_plant_text}, stat:{main_stat_value}, sample:{main_samp_value}, note:{main_note_text}")
            assert len(main_stat_value) == 4, f"蕾, 花, 実, 胞子の読み込みに失敗しています: {str(main_stat_value)}"
            assert len(main_samp_value) == 1, f"採種の読み込みに失敗しています: {str(main_samp_value)}"
            record = YasouRecord(index=row_index, plant_name=main_plant_text, stat=main_stat_value, sample=main_samp_value[0], note=main_note_text)
            report_info.records.append(record)
        except:
            logger.warning(f"skip index={row_index}")

    debugImgWrite(debug_stat_img, inspect.currentframe().f_code.co_name, "statImg")
    debugImgWrite(debug_stat_masked_img, inspect.currentframe().f_code.co_name, "statMaskedImg")

    return report_info  


def output_csv_report(report_info: YasouReportInfo, csvfile:str):
    logger.info(f"csvに出力 {csvfile}")
    with open(csvfile, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([str(report_info.date_year)+str(report_info.date_month).zfill(2)+str(report_info.date_day).zfill(2)])
        writer.writerow([report_info.member])
        writer.writerow([report_info.course_name, report_info.course_page])
        writer.writerow(["No","区間","種名","","蕾","花","実","胞子","採種","備考"])
        for record in report_info.records:
            try:
                writer.writerow([record.index, report_info.course_name, record.plant_name, "", YasouRecord.YesNoMark(record.stat_tubomi),YasouRecord.YesNoMark(record.stat_flower),YasouRecord.YesNoMark(record.stat_seed),YasouRecord.YesNoMark(record.stat_houshi),YasouRecord.YesNoMark(record.sample), record.note])
            except Exception as e:
                logger.error(f"書き込みエラーが発生しました: {e}")

# main
def main():
    global g_skipText
    global g_forceDate      # 明示的に指定した調査月
    global g_forceCourse    # 明示的に指定した調査コース
    global g_allOutput      # 読み込んだファイル順に1つのファイルにまとめて出力する
        
    oldMarkParser = False

    # ログを設定
    setup_logger()
     
    # 解析フレームのテンプレートファイル名
    frame_template = "frame_config.json"

    # 引数の読み取り処理
    parser = argparse.ArgumentParser()
    parser.add_argument('--skipText', '-s', action='store_true', help='テキスト認識をスキップする')
    parser.add_argument('--date', '-d', help='読み取り年月を明示的に指定する')
    parser.add_argument('--course', '-c', help='調査コースを明示的に指定する')
    parser.add_argument('--all', '-a', action='store_true', help='読み込んだファイル順に1つのファイルにまとめて出力する')
    parser.add_argument('--oldMarkParser', '-o', action='store_true', help='古いマークのパース方式を使う')
    parser.add_argument('--create-schema', '-x', action='store_true', help='スキーマを作成する')
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()

    files = args.files
    g_skipText = args.skipText
    g_forceDate = args.date
    g_forceCourse = args.course
    g_allOutput = args.all
    oldMarkParser = args.oldMarkParser
    create_schema = args.create_schema

    debugTmpImgRemove()
    # files = ["./record/202403/202403B01.JPG", "./record/202403/202403B02.JPG"]

    report_list:Dict[str, List[YasouReportInfo]] = dict()
    folder = os.path.dirname(files[0])

    total_page = 0
    for file in files:
        logger.info(f"読み込み処理開始:{file}")
        img = trim_report_image(file)
        if (img is None):
            continue
            
        report = scan_report(img, file, oldMarkParser)
        if (report is None):
            continue

        course_name = report.course_name
        total_page += 1
        if (g_allOutput):
            course_name = "All"
            report.course_page = total_page

        # コース別にレポートのリストを作成する
        if (course_name in report_list):
            report_list[course_name].append(report)
        else:
            report_list[course_name] = [report]

        logger.info(f"読み込み処理終了:{file}")


    for course_reports in report_list.values():
        course_reports.sort(key=lambda report: report.course_page)
        marged_report:YasouReportInfo = course_reports[0]
        marged_pages = f"{course_reports[0].course_page}"
        for report in course_reports[1:]:
            marged_report.records.extend(report.records)
            marged_pages += f",{report.course_page}"
            if (marged_report.member == ""):
                marged_report.member = report.member

        marged_report.course_page = marged_pages
        csvfile = os.path.join(folder, f"{marged_report.date_year:04}{marged_report.date_month:02}{marged_report.course_name}.csv")

        logger.info(f"CSVファイルに出力:{csvfile}")
        output_csv_report(marged_report, csvfile)


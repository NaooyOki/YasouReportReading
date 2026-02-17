import cv2
import numpy as np
import json
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Optional
import logging
import argparse

from scanreport.util import *
from scanreport.frame.frame_info import *
from scanreport.frame.table_util import *
from scanreport.frame.trim_report_frame import *
from scanreport.frame.text_scan import *


# ログ


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
    file_hander = logging.FileHandler('test.log', encoding='utf-8')
    file_hander.setLevel(logging.DEBUG)
    file_hander.setFormatter(formatter)
    logger.addHandler(file_hander)



if __name__ == "__main__":
    setup_logger()
    schema_file = "./template/PlantsInspectReport_schema.json"

    # 引数を解釈
    parser = argparse.ArgumentParser()
    parser.add_argument('--create-schema', '-c', action='store_true', help='スキーマを作成する')
    args = parser.parse_args()

    if args.create_schema:
        # テンプレート画像を読み込む
        template_file = "./template/PlantsInspectReport_v1.1.jpg"
        template_img = cv2.imread(template_file)
        template_img = trim_paper_frame(template_img)
        template_img, found = trim_inner_mark2(template_img)
        timg_h, timg_w = template_img.shape[:2]

        # テンプレートから表情報を抽出する
        template_img_frame = create_line_mask(template_img)
        debugImgWrite(template_img_frame, "test", "2template_mask")
        template_tables = analyze_table_structure_as_struct(template_img_frame)
        logger.info(f"表の数: {len(template_tables)}")

        schemas = []
        for i, table in enumerate(template_tables):
            logger.info(f"表{i}: {table}")
    
            # スキーマを作る
            schema = TableSchema.create_schema_from_table(table, (timg_w, timg_h), f"Table{i}")
            logger.info(f"スキーマ: {schema}")
            schemas.append(schema)

        # スキーマをシリアライズする
        json_outputs = []
        for schema in schemas:
            json_output = schema.to_json()
            logger.info(f"JSON保存: {json_output}")
            json_outputs.append(json_output)
        
        # 
        # スキーマ情報をJSONファイルに保存
        with open(schema_file, "w", encoding="utf-8") as f:
            # リスト形式で保存
            json.dump([asdict(s) for s in schemas], f, ensure_ascii=False, indent=4)
        logger.info(f"スキーマを {schema_file} に保存しました。")
        exit(0)


    # テスト用の画像を読み込む
    testimg_file = "./record/BadData/24021839.JPG"
    img = cv2.imread(testimg_file)
    img = trim_paper_frame(img)
    img, found = trim_inner_mark2(img)
    img_h, img_w = img.shape[:2]

    # スキーマーを読み込む
    schemas = []
    with open(schema_file, "r", encoding="utf-8") as f:
        json_schemas = json.load(f)
        for s in json_schemas:
            schema = TableSchema.from_json(s)
            schemas.append(schema)

    # 帳票を作成する
    sheet = RecordSheet.create_sheet_from_img(img, schemas)
    sheet.match_tables()
    debugImgWrite(sheet.img, "test", "1org_img")

    # OCR機能を使ってテキスト情報を読み取る
    img_reader = VisonImgTextReader()
    textCacheFile = "./cache/" + os.path.basename(testimg_file) + ".pickle"     # テキスト情報をキャッシュするファイル
    readCache = False
    if (os.path.exists(textCacheFile)):
        if (os.path.getmtime(textCacheFile) > os.path.getmtime(testimg_file)):
            readCache = True
    if (readCache):
        logger.debug(f"read cache from {textCacheFile}")
        img_reader.load_file(textCacheFile)
    else:
        logger.debug("scan and create cache")
        img_reader.read_image(img)
        img_reader.save_file(textCacheFile)

    # ヘッダを得る
    header = sheet.matched_tables["Table0"]
    if (header is not None):
        logger.info(f"header: {header}")
        date_cell = header.get_cell(row=0, col=0)
        date_text = img_reader.extract_text_from_region2(date_cell.rect)
        logger.info(f"date_cell: {date_cell}. value={date_text}")
        weather_cell = header.get_cell(0, 1)
        weather_text = img_reader.extract_text_from_region2(weather_cell.rect)
        logger.info(f"weather_cell: {weather_cell}. value={weather_text}")
        member_cell = header.get_cell(1, 0)
        member_text = img_reader.extract_text_from_region2(member_cell.rect)
        logger.info(f"member_cell: {member_cell}, value={member_text}")

    # コース情報を得る
    course = sheet.matched_tables["Table1"]
    if (course is not None):
        logger.info(f"course: {course}")
        route_cell = course.get_cell(0, 0)
        route_text = img_reader.extract_text_from_region2(route_cell.rect)
        logger.info(f"route_cell: {route_cell}, value={route_text}")
        num_cell = course.get_cell(0, 1)
        num_text = img_reader.extract_text_from_region2(num_cell.rect)
        logger.info(f"num_cell: {num_cell}, value={num_text}")



    
    exit(0)

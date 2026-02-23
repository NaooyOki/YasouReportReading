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

from dataclasses import dataclass, field, asdict
from typing import ClassVar
from dataclasses_json import dataclass_json, config
from marshmallow import Schema, fields


from ..mark.status_train_info import *
from ..util import *
from ..frame.trim_report_frame import *
from ..frame.frame_info import *
from ..frame import *
from ..mark.mark_parser import *
from ..frame.table_util import *


logger = logging.getLogger(__name__)


def create_traindata_from_report(target_file: str) -> StatReportInfo:
    report_info = StatReportInfo(file_name=target_file)

    logger.info(f"読み込み処理開始:{target_file}")
    trim_img2 = read_imagefile_with_trim(target_file)
    if trim_img2 is None:
        logger.error(
            f"サポートしていない画像ファイルのため、学習データ作成をスキップします:{target_file}"
        )
        return None

    # 画像をグレースケールにして白黒反転する
    img_gray = cv2.cvtColor(trim_img2, cv2.COLOR_BGR2GRAY)
    ret, scan_image = cv2.threshold(
        img_gray, 130, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

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

    # メイン部分を取り出す
    main = sheet.matched_tables["Main"]
    if main is None:
        logger.error("メイン部分が見つかりません")
        return None

    # 蕾花実列用のパーサーを作成
    main_stat_header = main.get_cell(0, 2)
    stats_header_image = sheet.get_image(main_stat_header.rect)
    stat_parser = MarkListImageParser()
    stat_parser.readMarkHeaderImage(stats_header_image, verify_num=4)
    report_info.header.stat_image = stat_parser.headerListImage

    # 採取列用のパーサーを作成
    main_samp_header = main.get_cell(0, 3)
    samp_header_image = sheet.get_image(main_samp_header.rect)
    samp_parser = MarkListImageParser()
    samp_parser.readMarkHeaderImage(samp_header_image, verify_num=1)
    report_info.header.sample_image = samp_parser.headerListImage

    # メイン部分
    for row_index in range(1, main.row_count):
        # main_no = main.get_frame_in_cluster(row_index, 0)
        # main_plant = main.get_frame_in_cluster(row_index, 1)

        main_stat_header = main.get_cell(0, 2)
        stats_header_image = sheet.get_image(main_stat_header.rect)

        main_stat = main.get_cell(row_index, 2)
        stat_image = sheet.get_image(main_stat.rect)
        detected_stat = stat_parser.readMarkListImage(stat_image)
        main_stat.value = detected_stat

        main_sample = main.get_cell(row_index, 3)
        samp_image = sheet.get_image(main_sample.rect)
        detected_samp = samp_parser.readMarkListImage(samp_image)
        main_sample.value = detected_samp

        # main_note = main.get_frame_in_cluster(row_index, 4)
        try:
            assert (
                len(detected_stat) == 4
            ), f"蕾, 花, 実, 胞子の読み込みに失敗しています: {str(detected_stat)}"
            assert (
                len(detected_samp) == 1
            ), f"採種の読み込みに失敗しています: {str(detected_samp)}"
            # print(f"No:{row_index}, stat_img:{stat_image.shape}, stat:{main_stat.value}, sample:{main_sample.value}")

            record = StatRecord(
                index=row_index,
                stat_tubomi=detected_stat[0],
                stat_flower=detected_stat[1],
                stat_seed=detected_stat[2],
                stat_houshi=detected_stat[3],
                sample=detected_samp[0],
            )
            record.stat_image = stat_parser.markListImage
            record.sample_image = samp_parser.markListImage
            report_info.records.append(record)
        except:
            print(f"skip index={row_index}")

    return report_info


def find_jpg_files(directory_path) -> List[str]:
    """
    指定されたディレクトリ以下のJPGファイルのパスを取得する関数
    Args:
      directory_path: 検索対象のディレクトリのパス
    Returns:
      JPGファイルのパスのリスト
    """
    try:
        pattern = os.path.join(directory_path, "**", "*.jpg")
        jpg_files = glob.glob(pattern, recursive=True)
    except Exception as e:
        print(f"Error: {e}")
        jpg_files = []
    return jpg_files


# main
g_skipText = False


def make_train_main(folder_path):
    """
    指定したフォルダ以下の画像データを再帰的に読み込んで、学習用のデータを作成する関数
    """

    files = findFilesRecursive(folder_path, "*.jpg")
    logger.info(f"top: {folder_path} files: {files}")

    debugTmpImgRemove()
    # files = ["./record/202403/202403B01.JPG", "./record/202403/202403B02.JPG"]

    for file in files:
        logger.info(f"読み込み処理開始:{file}")
        report = create_traindata_from_report(file)
        logger.info(f"読み込み処理終了:{file}")
        if report is not None:
            filename, ext = os.path.splitext(os.path.basename(file))
            report.write_file("./stat_img/", filename)

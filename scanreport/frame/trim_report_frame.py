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

from ..util import *

# ログを使う
logger = logging.getLogger(__name__)


def trim_paper_frame(img: np.ndarray) -> np.ndarray:
    """
    黒地に映ったA4用紙の輪郭を検出する
    @param img:image 暗い背景に写された白い記録用紙の写真
    @return :image 記録用紙をトリミングした画像イメージ
      記録用紙をスキャナーで読み取った場合は暗い背景がないので、元の画像イメージをそのまま返す
    """
    img_debug = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 画像処理のノイズ除去
    ret, img_gray2 = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    debugImgWrite(img_debug, inspect.currentframe().f_code.co_name, "1input")

    # 境界線を見つける
    contours, hierarchy = cv2.findContours(
        img_gray2,
        cv2.RETR_EXTERNAL,  # 一番外側の輪郭のみ
        cv2.CHAIN_APPROX_SIMPLE,  # 輪郭座標の省略
    )
    for i, contour in enumerate(contours):
        # 傾いていない外接する矩形領域を求める
        x, y, w, h = cv2.boundingRect(contour)

        # 画像の50%以上の大きさを占める輪郭をA4用紙の輪郭と扱う
        # print(img_gray.shape)
        if (w > img_gray2.shape[1] * 0.5) and (h > img_gray2.shape[0] * 0.5):
            # 領域の近似形状を求める
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) < 4:
                continue
            (top_left, top_right, bot_left, bot_right) = calcRectEdge(contour)
            trim_img = trimAsRectangle(img, top_left, top_right, bot_left, bot_right)
            if __debug__:
                cv2.drawContours(img_debug, contours, i, (255, 0, 0), 2)
                cv2.rectangle(img_debug, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
                cv2.circle(img_debug, top_left, 20, (0, 255, 255), -1)
                cv2.circle(img_debug, top_right, 20, (0, 255, 255), -1)
                cv2.circle(img_debug, bot_left, 20, (0, 255, 255), -1)
                cv2.circle(img_debug, bot_right, 20, (0, 255, 255), -1)
                debugImgWrite(
                    img_debug, inspect.currentframe().f_code.co_name, "2output"
                )
            return trim_img

    # 　見つからなかった場合は、用紙全体を写したか、白地の背景と解釈して、イメージ領域をそのまま返す
    logger.warning("Paper frame not found. Returned whole picture as a paper.")
    return img


def trim_inner_mark(img: np.ndarray) -> np.ndarray:
    """
    四隅の位置合わせマークで囲まれた区画を抽出し、台形補正する
    位置合わせマークは、３つの三角形と１つの正方形(右下)
    @param img: 元の画像(白地が背景)
    @return: 抽出された画像
    """
    # 白地のA4用紙のレポートから、▲と■マークを検出する
    img_debug = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 画像処理のノイズ除去
    ret, img_gray2 = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    debugImgWrite(img_gray2, inspect.currentframe().f_code.co_name, "1gray")

    # 境界線を見つける
    contours, hierarchy = cv2.findContours(
        img_gray2,
        cv2.RETR_EXTERNAL,  # 2段階の輪郭で取得。マークには子供がいないはず
        cv2.CHAIN_APPROX_SIMPLE,
    )

    triangles = 0
    box = 0
    tri_area = []
    box_area = []

    # マークの推定サイズ
    img_w = img_gray2.shape[1]
    img_h = img_gray2.shape[0]
    mark_w = 6 / 210 * img_w
    mark_h = 6 / 297 * img_h
    min_mark_w = mark_w * 0.75
    max_mark_w = mark_w * 1.25
    min_mark_h = mark_h * 0.75
    max_mark_h = mark_h * 1.25

    for i, contour in enumerate(contours):
        # 小さな領域はノイズとみなして、スキップする
        area = cv2.contourArea(contour)
        # print(f"area = {area}")
        if area < min_mark_w * min_mark_h * 0.5:
            # print(f" detect edge mark: skiped small area size={area}")
            continue
        elif area > max_mark_w * max_mark_h:
            # print(f" detect edge mark: skiped big area size={area}")
            continue

        # 領域の形状を見つける
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 3:
            # 塗りつぶした三角形
            cv2.drawContours(img_debug, [approx], 0, (0, 255, 255), 5)
            cv2.putText(
                img_debug,
                f"size={area}",
                approx[0][0],
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0),
            )
            triangles += 1
            tri_area.append(approx)
        elif len(approx) == 4:
            # 塗りつぶした４角形かを確認
            cv2.drawContours(img_debug, [approx], 0, (0, 255, 0), 5)
            cv2.putText(
                img_debug,
                f"size={area}",
                approx[0][0],
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0),
            )
            box += 1
            box_area.append(approx)
        else:
            cv2.drawContours(img_debug, [approx], 0, (0, 0, 255), 5)
            cv2.putText(
                img_debug,
                f"size={area}, edge={len(approx)}",
                approx[0][0],
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0),
            )

    found = False
    if (len(tri_area) == 3) and (len(box_area) == 1):
        # 四隅を発見できた
        center = (img.shape[1] / 2, img.shape[0] / 2)
        right_bot = min(box_area[0], key=lambda x: np.linalg.norm(x[0] - center))[
            0
        ]  # 四角形は右下固定
        tri = []
        for t in tri_area:
            tri.append(min(t, key=lambda x: np.linalg.norm(x[0] - center))[0])

        # 三つの三角について、右下からの距離でソートして、遠い順に左上、右上、左下を求める。
        (left_bot, right_top, left_top) = sorted(
            tri, key=lambda x: np.linalg.norm(x - right_bot)
        )
        new_img = trimAsRectangle(img, left_top, right_top, left_bot, right_bot)

        cv2.putText(
            img_debug,
            "left_top",
            left_top + (0, -20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (0),
        )
        cv2.putText(
            img_debug,
            "right_top",
            right_top + (0, -20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (0),
        )
        cv2.putText(
            img_debug, "left_bot", left_bot + (0, 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0)
        )
        cv2.putText(
            img_debug,
            "right_bot",
            right_bot + (0, 20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (0),
        )
        points = np.array([left_top, right_top, right_bot, left_bot]).reshape(1, -1, 2)
        cv2.polylines(img_debug, points, isClosed=True, color=(255, 0, 0), thickness=5)
        debugImgWrite(img_debug, inspect.currentframe().f_code.co_name, "2box")
        found = True
    else:
        # 四隅が見つからない場合は、イメージをそのまま返す
        logger.warning(
            f"四隅のマークを検出できませんでした。元のイメージをそのまま返します。(三角形:{len(tri_area)}, 四角形:{len(box_area)})"
        )
        found = False
        new_img = img

    debugImgWrite(img_debug, inspect.currentframe().f_code.co_name, "2box")
    return (new_img, found)


def trim_inner_mark2(img: np.ndarray) -> np.ndarray:
    """
    四隅の位置合わせマークで囲まれた区画を抽出し、台形補正する
    位置合わせマークは、３つの中空の正方形と１つの塗りつぶされた正方形(右下)
    @param img: 元の画像(白地が背景)
    @return: 抽出された画像
    """
    # 白地の画像なので、反転させて処理する
    img_debug = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 画像処理のノイズ除去
    ret, img_inv = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    debugImgWrite(img_inv, inspect.currentframe().f_code.co_name, "1gray")

    # 境界線を見つける
    contours, hierarchy = cv2.findContours(
        img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE  # 外輪だけを抽出する
    )

    anker = 0
    box = 0
    anker_area = []
    box_area = []

    # マークの推定サイズ
    img_w = img_inv.shape[1]
    img_h = img_inv.shape[0]
    mark_w = 110 / 2550 * img_w
    mark_h = 110 / 3300 * img_h
    min_mark_w = mark_w * 0.5
    max_mark_w = mark_w * 1.25
    min_mark_h = mark_h * 0.5
    max_mark_h = mark_h * 1.25

    for i, contour in enumerate(contours):
        # マークと思われる領域サイズ以外はスキップする
        rect = cv2.minAreaRect(contour)
        center, (width, height), angle = rect
        # print(f"detect area = {rect}")
        if (width < min_mark_w) or (height < min_mark_h):
            # print(f" detect edge mark: skiped small area rect={rect}")
            continue
        elif (width > max_mark_w) or (height > max_mark_h):
            # print(f" detect edge mark: skiped big area size={rect}")
            continue

        # 領域の形状をチェックする
        # print(f" check edge mark: rect={rect}")
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            # マークの形状をチェックする
            x, y, w, h = cv2.boundingRect(approx)
            cropped_image = img_inv[y : y + h, x : x + w]
            sub_contours, hierarchy2 = cv2.findContours(
                cropped_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            # 大きな内部領域だけをカウントする
            subrect_num = 0
            for j, sub_contour in enumerate(sub_contours):
                sub_area = cv2.contourArea(sub_contour)
                # print(f"sub_area_ratio = {sub_area/(w*h)*100.0}%")
                if sub_area < w * h * 0.09:
                    continue
                subrect_num += 1
                sub_rect = cv2.boundingRect(sub_contour)
                cv2.rectangle(
                    img_debug,
                    (x + sub_rect[0], y + sub_rect[1]),
                    (x + sub_rect[0] + sub_rect[2], y + sub_rect[1] + sub_rect[3]),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    img_debug,
                    f"subrect{j}",
                    (x + sub_rect[0], y + sub_rect[1] - 20),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (0),
                )

            logger.debug(f"subrect_num = {subrect_num}")

            if subrect_num == 1:
                # 塗りつぶされたマーク(右下)
                anker += 1
                anker_area.append(approx)
                cv2.drawContours(img_debug, [approx], 0, (255, 0, 0), 5)
                cv2.putText(
                    img_debug,
                    f"rect={rect}",
                    approx[0][0],
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (0),
                )
            elif subrect_num == 2:
                # 真ん中が空いたマーク(左上、右上、左下のどれか)
                box += 1
                box_area.append(approx)
                cv2.drawContours(img_debug, [approx], 0, (0, 255, 0), 5)
                cv2.putText(
                    img_debug,
                    f"rect={rect}",
                    approx[0][0],
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (0),
                )
            else:
                cv2.drawContours(img_debug, [approx], 0, (0, 0, 128), 5)
                cv2.putText(
                    img_debug,
                    f"rect={rect}, edge={len(approx)}",
                    approx[0][0],
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (0),
                )
                logger.debug(
                    f"skiped wrong mark: edge={len(approx)}, rect={rect}, sub_contours={subrect_num}"
                )

        else:
            cv2.drawContours(img_debug, [approx], 0, (0, 0, 255), 5)
            cv2.putText(
                img_debug,
                f"rect={rect}, edge={len(approx)}",
                approx[0][0],
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0),
            )
            logger.debug(f"skiped edge mark: edge={len(approx)}, rect={rect}")

    if (len(anker_area) == 1) and (len(box_area) == 3):
        # 四隅を発見できた
        found = True
        center = (img.shape[1] / 2, img.shape[0] / 2)
        right_bot = min(anker_area[0], key=lambda x: np.linalg.norm(x[0] - center))[
            0
        ]  # 四角形は右下固定
        tri = []
        for t in box_area:
            tri.append(min(t, key=lambda x: np.linalg.norm(x[0] - center))[0])

        # 三つのマーカーについて、右下からの距離でソートして、遠い順に左上、右上、左下を求める。
        (left_bot, right_top, left_top) = sorted(
            tri, key=lambda x: np.linalg.norm(x - right_bot)
        )

        # 台形補正する
        new_img = trimAsRectangle(img, left_top, right_top, left_bot, right_bot)

        cv2.putText(
            img_debug,
            "left_top",
            left_top + (0, -20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (0),
        )
        cv2.putText(
            img_debug,
            "right_top",
            right_top + (0, -20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (0),
        )
        cv2.putText(
            img_debug, "left_bot", left_bot + (0, 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0)
        )
        cv2.putText(
            img_debug,
            "right_bot",
            right_bot + (0, 20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (0),
        )
        points = np.array([left_top, right_top, right_bot, left_bot]).reshape(1, -1, 2)
        cv2.polylines(
            img_debug, points, isClosed=True, color=(255, 255, 0), thickness=5
        )
        debugImgWrite(img_debug, inspect.currentframe().f_code.co_name, "2box")
    else:
        # 四隅が見つからない場合は、イメージをそのまま返す
        found = False
        logger.warning(
            f"四隅のマークを検出できませんでした。元のイメージをそのまま返します。(box:{len(box_area)}, anker:{len(anker_area)})"
        )
        new_img = img

    debugImgWrite(img_debug, inspect.currentframe().f_code.co_name, "2box")
    return (new_img, found)


def read_imagefile_with_trim(file: str) -> np.ndarray:
    """
    画像ファイルを読み込み、用紙の輪郭検出および四隅のマークに基づいたトリミングと台形補正を行う。
    @param file: 読み取り対象の画像ファイルパス
    @return: 補正・トリミング済みの画像イメージ。マークが見つからない場合はNoneを返す。
    """
    # 画像を読み込む
    img = cv2.imread(file)
    if img is None:
        logger.error(f"画像の読み込みに失敗しました: {file}")
        return None

    # 暗い背景から紙の白い部分を切り抜く
    trim_img = trim_paper_frame(img)

    # 四隅のマークで記録部分を切り出す
    trim_img2, found = trim_inner_mark2(trim_img)
    if found:
        return trim_img2

    trim_img2, found = trim_inner_mark(trim_img)
    if found:
        return trim_img2

    logger.warning(
        f"四隅のマークが見つかりませんでした。このファイルの解析をスキップします。: {file}"
    )
    return None

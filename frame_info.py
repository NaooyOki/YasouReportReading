import cv2
import numpy as np
import math
import json
import glob
import csv
import os
import sys
import utility as util
import inspect
from dataclasses import dataclass, field, asdict
from typing import ClassVar
from dataclasses_json import dataclass_json, config
from marshmallow import Schema, fields
from typing import MutableMapping
from typing import List
import re
import bisect

#@dataclass_json
#@dataclass
class Frame:
    """
    フレームの情報。フレームとは、複数のセルと縦と横のクラスター情報から構成される。
    矩形の領域情報は[x, y, w, h]で、親フレームを基準にした相対座標
    """
    # 
    #name: str
    #rect: tuple[int, int, int, int]     # 自身の矩形の領域  [x, y, w, h]  親フレームから見た時の座標
    #parent:type(FrameInfo2) = None
    #children:dict = MutableMapping[str, type(__class__)]   # field(default_factory=dict)
    #col_cluster_list: list[ClusterInfo] = field(default_factory=list)
    #row_cluster_list: list[ClusterInfo] = field(default_factory=list)

    FORMAT_FRAME = "frame"
    FORMAT_TEXT = "text"
    FORMAT_MARK = "mark"
    SCHEMA_ALL_TEXT = r'(.*)'
    SCHEMA_DATE_TEXT = r'(\d*)年(\d*)月(\d*)日'

    def __init__(self, name, rect, parent=None) -> None:
        self.name = name
        self.rect = rect
        self.parent = parent
        self.children = dict()
        if (parent is not None):
            parent.children[self.name] = self
        self.format = Frame.FORMAT_FRAME
        self.value = None
        self.schema = None
        self.cluster_list_row = []
        self.cluster_list_col = []

    def abs_rect(self) -> list[int]:
        rect = self.rect.copy()
        parent = self.parent
        while (parent is not None):
            rect[0] += parent.rect[0]
            rect[1] += parent.rect[1]
            parent = parent.parent
        return rect 

    def to_dict(self):
        dic = {
            "name": self.name,
            "rect": self.rect,
            "format": self.format,
            "value": self.value,
            "schema": self.schema,
                }
        if (len(self.children) > 0):
            dic["children"] = list(map(lambda c: c.to_dict(), self.children.values()))
            dic["cluster_list_col"] = list(map(lambda c: c.to_dict(), self.cluster_list_col))
            dic["cluster_list_row"] = list(map(lambda c: c.to_dict(), self.cluster_list_row))

        return dic

    def scan_value(self, scan):
        ptn = re.compile(self.schema)
        text = scan
        if result := ptn.search(text):
            self.value = result.group(0)
        else:
            self.value = ""
        return self.value
    
    
    def scan_text(self) -> str:
        return "test"


    def create_claster_list_sub(self, direction):
        # 子輪郭からクラスター情報を作る
        if (len(self.children) > 0):
            if (direction == Cluster.DIRECT_ROW):
                cluster_name = "row_"
                pos_index = 1
                len_index = 3
                subpos_index = 0
            else:
                cluster_name = "col_"
                pos_index = 0
                len_index = 2
                subpos_index = 1

            cluster_list = []
            for child in self.children.values():
                found = False
                for cluster in cluster_list:
                    if (cluster.is_same_cluster(child.rect)):
                        # 同じ行/列クラスターに所属する子フレームが見つかったので、列/行の位置の順にソートしてリストに加える
                        bisect.insort(cluster.frames, child.name, key=lambda name: self.children[name].rect[subpos_index])
                        found = True
                        break
                if (not found):
                    # どの行クラスーにも所属しない子フレームのため、新しい行クラスターを作って、列の位置の順にソートして追加する
                    new_cluster = Cluster(f"{cluster_name}{len(cluster_list)+1}", Cluster.DIRECT_ROW, child.rect[pos_index], child.rect[len_index])
                    new_cluster.frames.append(child.name)
                    bisect.insort(cluster_list, new_cluster, key=lambda c: c.pos)
            
            return cluster_list


    def create_claster_list(self):
        self.cluster_list_row = self.create_claster_list_sub(Cluster.DIRECT_ROW)
        self.cluster_list_col = self.create_claster_list_sub(Cluster.DIRECT_COL)
        return

    def get_frame_in_cluster(self, col_index, row_index):
        try:
            cluster = self.cluster_list_row[col_index]
            frame_name = cluster.frames[row_index]
            frame = self.children[frame_name]
        except IndexError:
            print(f"範囲外のインデクスにアクセスしました col={col_index}, row={row_index}")
            return None
        return frame
class Cluster:
    """
    子フレームの集合を列または行のあつまりで扱うためのクラス
    """
    # クラスターの方向
    DIRECT_COL = 'col'  
    DIRECT_ROW = 'row'
    NEAR_VALUE = 5 # 1%の違い以内なら、同じ位置や長さとみなす

    def __init__(self, name, direct, pos, len) -> None:
        self.name = name
        self.direct = direct
        self.pos = pos
        self.len = len
        self.frames = []

    def is_same_cluster(self, rect):
        is_same = False
        if (self.direct == Cluster.DIRECT_COL):
            is_same = ((math.isclose(self.pos, rect[0], abs_tol=Cluster.NEAR_VALUE)) and (math.isclose(self.len, rect[2], abs_tol=Cluster.NEAR_VALUE)))
        else:
            is_same = ((math.isclose(self.pos, rect[1], abs_tol=Cluster.NEAR_VALUE)) and (math.isclose(self.len, rect[3], abs_tol=Cluster.NEAR_VALUE)))
        return is_same
    
    def to_dict(self):
        return {"direct": self.direct, "pos": self.pos, "len": self.len, "frames": self.frames}


class FrameInfo3:
    # フレームの情報。フレームとは、複数のセルと縦と横のクラスター情報から構成される。
    def __init__(self, name, rect, parent=None) -> None:
        self.name = name
        self.rect = rect
        self.parent = parent
        self.children = dict()
        self.size = None
        if (not parent is None):
            parent.children[name] = self

    def abs_rect(self) -> list[int]:
        rel_rect = self.rect.copy()
        parent = self.parent
        root = self
        while (not parent is None):
            root = parent
            rel_rect[0] = rel_rect[0] * parent.rect[2] + parent.rect[0]
            rel_rect[1] = rel_rect[1] * parent.rect[3] + parent.rect[1]
            rel_rect[2] = rel_rect[2] * parent.rect[2]
            rel_rect[3] = rel_rect[3] * parent.rect[3]
            parent = parent.parent
        rect = rel_rect
        print(f"rel_rect = {rel_rect}")
        rect[0] *= root.size[0]
        rect[1] *= root.size[1]
        rect[2] *= root.size[0]
        rect[3] *= root.size[1]
        return rect 
            

    def to_dict(self):
        children = dict()
        for name in self.children:
            child = self.children[name]
            children[name] = child.to_dict()
        return {
            "name": self.name,
            "rect": self.rect,
            "children": children
                }


from typing import List
def get_frames(img) -> List[Frame]:
    img_debug = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_gray2 = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    util.debugImgWrite(img_debug, inspect.currentframe().f_code.co_name, "1input")

    # 入力された領域をRootフレームとする
    img_w = img_gray2.shape[1]
    img_h = img_gray2.shape[0]
    root_frame = Frame("root", [0, 0, img_w, img_h], None)
    
    # 輪郭のツリーを見つける
    contours, hierarchy = cv2.findContours(
        img_gray2, 
        cv2.RETR_TREE, 
        cv2.CHAIN_APPROX_SIMPLE
        ) 
    
    # 第一階層の輪郭を追って情報を得る
    cont_index = 0
    while (cont_index != -1):
        get_frames_sub(img_gray2, contours, hierarchy, cont_index, 0, root_frame, img_debug)
        cont_index = hierarchy[0][cont_index][0]   # 次の領域

    root_frame.create_claster_list()


    util.debugImgWrite(img_debug, inspect.currentframe().f_code.co_name, "2output")
    return (root_frame)

def get_frames_sub(image, contours, hierarchy, target_cont_index, level, parent, img_debug):
    """
    入れ子になった矩形領域をFrameクラスのインスタンスとして階層的に抽出する
    Args:
        image: 入力画像
        contours: 輪郭のリスト
        hierarchy: 輪郭の階層情報
        target_cont_index: 対象の輪郭のインデックス
        level: 階層レベル
        parent: 親のFrameクラス
    """

    trim_offset = 10
    color = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
    skip_color = (0, 0, 128)

    # 輪郭のインデックスが無効の場合は、Noneを返す
    if target_cont_index == -1:
        return None

    # 2階層を超えた場合は、Noneを返す
    if level >= 2:
        return None
    
    # 自身の領域情報を得る
    contour = contours[target_cont_index]
    area = cv2.contourArea(contour)
    x,y,w,h = cv2.boundingRect(contour)
    img_w = image.shape[1]
    img_h = image.shape[0]
    x_rel = x / img_w
    y_rel = y / img_h
    w_rel = w / img_w
    h_rel = h / img_h
    
    # 小さい領域は無視する
    if ((w_rel < 0.02) or (h_rel < 0.02)):
        print(f"skipped small area {x},{y},{w},{h}")
        cv2.rectangle(img_debug, (x, y), (x+w, y+h), skip_color, trim_offset)
        return None
    
    # 自分自身のFrameクラスを作る
    frame = Frame(f"frame_{target_cont_index}", [x - parent.rect[0], y - parent.rect[1], w, h], parent)
    cv2.rectangle(img_debug, (x, y), (x+w, y+h), color[level], trim_offset)
    cv2.putText(img_debug, f"frame: {frame.name}", (x, y-20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0))

    # 子輪郭の情報リストを得る
    child_cont_index = hierarchy[0][target_cont_index][2]
    while (child_cont_index != -1):
        get_frames_sub(image, contours, hierarchy, child_cont_index, level + 1, frame, img_debug)
        child_cont_index = hierarchy[0][child_cont_index][0]   # 次の領域

    # 子輪郭からクラスター情報を作る
    frame.create_claster_list()

    return frame


import trim_report_frame
import text_scan
def scan_frame(frame:Frame, scanner:text_scan.VisonImgTextReader):
    if (len(frame.children) > 0):
        for child in frame.children.values():
            scan_frame(child, scanner)
    else:
        rect = frame.abs_rect()
        text = scanner.extract_text_from_region(rect[0], rect[1], rect[2], rect[3])
        frame.value = text
        print(f"scanned text:{text} from {frame.name}:{rect}")

# main

if __name__ == '__main__':
    test_file = "./record/202403/202403B02.JPG"
    #test_file = "./template/PlantsInspectReport_v1.1.jpg"
    cache_file = "./cache/" + os.path.basename(test_file) + ".pickle"
    img = cv2.imread(test_file)
    trim_img = trim_report_frame.trim_paper_frame(img)
    trim_img2 = trim_report_frame.trim_inner_mark2(trim_img)

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
    debug_img = trim_img2.copy()
    root = get_frames(trim_img2)

    # 取得したフレーム内の文字を読み込む
    scan_frame(root, img_reader)

    # ヘッダを取り出す
    head = root.get_frame_in_cluster(0, 0)
    head_date = head.get_frame_in_cluster(0, 0)
    print(f"head_date: {head_date.value}")
    head_member = head.get_frame_in_cluster(1, 0)
    print(f"head_member: {head_member.value}")
    head_wed = head.get_frame_in_cluster(0, 1)
    print(f"head_wed: {head_wed.value}")
    
    # コースを取り出す
    course = root.get_frame_in_cluster(1, 0)
    course_route = course.get_frame_in_cluster(0, 0)
    print(f"course_route: {course_route.value}")
    course_page = course.get_frame_in_cluster(0, 1)
    print(f"course_page: {course_page.value}")

    # メイン部分を取り出す
    main = root.get_frame_in_cluster(2, 0)
    for row_index in range(len(main.cluster_list_row)):
        main_no = main.get_frame_in_cluster(row_index, 0)
        main_plant = main.get_frame_in_cluster(row_index, 1)
        main_kind = main.get_frame_in_cluster(row_index, 2)
        main_sample = main.get_frame_in_cluster(row_index, 3)
        main_note = main.get_frame_in_cluster(row_index, 4)
        try:
            print(f"No:{main_no.value}, plant:{main_plant.value}, kind:{main_kind.value}, sample:{main_sample.value}, note:{main_note.value}")
        except:
            print(f"skip index={row_index}")


    # JSONデータをファイルに書き込み
    d = root.to_dict()
    # print(f"root_to_dict={d}")
    with open("./tmp/frame_info.json", "w") as f:
        json.dump(d, f, indent=4)

    exit()



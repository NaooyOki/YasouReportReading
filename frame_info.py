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
import re

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
        self.children = []
        if (parent is not None):
            parent.children.append(self)
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
            dic["children"] = list(map(lambda c: c.to_dict(), self.children))
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
    
    def create_claster_list(self):
        # 子輪郭からクラスター情報を作る
        if (len(self.children) > 0):
            cluster_list_row = []
            for child in self.children:
                found = False
                for cluster in cluster_list_row:
                    if (cluster.is_same_cluster(child.rect)):
                        cluster.frames.append(child.name)
                        found = True
                        break
                if (not found):
                    new_cluster = Cluster(f"row_{len(cluster_list_row)+1}", Cluster.DIRECT_ROW, child.rect[1], child.rect[3])
                    new_cluster.frames.append(child.name)
                    cluster_list_row.append(new_cluster)
            self.cluster_list_row = cluster_list_row

            cluster_list_col = []
            for child in self.children:
                found = False
                for cluster in cluster_list_col:
                    if (cluster.is_same_cluster(child.rect)):
                        cluster.frames.append(child.name)
                        found = True
                        break
                if (not found):
                    new_cluster = Cluster(f"col_{len(cluster_list_col)+1}", Cluster.DIRECT_COL, child.rect[0], child.rect[2])
                    new_cluster.frames.append(child.name)
                    cluster_list_col.append(new_cluster)
            self.cluster_list_col = cluster_list_col

class Cluster:
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


# main
import trim_report_frame
if __name__ == '__main__':
    #test_file = "./record/202403/202403B01.JPG"
    test_file = "./template/PlantsInspectReport_v1.1.jpg"
    img = cv2.imread(test_file)
    trim_img = trim_report_frame.trim_paper_frame(img)
    trim_img2 = trim_report_frame.trim_inner_mark2(trim_img)
    

    debug_img = trim_img2.copy()
    root = get_frames(trim_img2)
    d = root.to_dict()
    print(f"root_to_dict={d}")

    # JSONデータをファイルに書き込み
    with open("./tmp/frame_info.json", "w") as f:
        json.dump(d, f, indent=4)
    exit()




    root = Frame("root", [0, 0, 1000, 2000], None)
    head = Frame("head", [100, 100, 400, 100], root)
    head_date = Cell("date", [0, 0, 300, 50], head, Cell.FORMAT_TEXT, "", Cell.SCHEMA_DATE_TEXT)
    head_wed = Cell("weadher", [300, 0, 100, 50], head, Cell.FORMAT_TEXT, "", Cell.SCHEMA_ALL_TEXT)
    head_mem = Cell("member", [0, 50, 400, 50], head, Cell.FORMAT_TEXT, "", Cell.SCHEMA_ALL_TEXT)
    page = Frame("page", [700, 100, 200, 100], root)
    main = Frame("main", [100, 250, 800, 1550], root)

    head_date.scan_value('日付： 2024年3月27日')
    head_wed.scan_value('天気: 晴れ')
    head_mem.scan_value('青木、石田、上本、遠藤、大分')


    head_wed_rect = head_wed.abs_rect()
    print(f"head_rect={head_wed_rect}")
    d = root.to_dict()
    print(f"root_to_dict={d}")
    exit()


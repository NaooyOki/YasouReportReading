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
from typing import MutableMapping
from typing import List, Dict, Optional
import re
import bisect
from .table_util import *
from .text_scan import VisonImgTextReader

from ..util import *

logger = logging.getLogger(__name__)


class RecordSheet:
    """
    帳票を表現するクラス。以下の情報から構成される。
    ・帳票の画像イメージ(白黒)
    ・複数の表の枠情報
    ・帳票のスキーマ
    ・画像イメージから読み取ったテキスト情報
    """

    tables:List[Table] = []
    matched_tables:Dict[str, Table] = {}
    schema:Schema = None
    img:np.ndarray = None


    def __init__(self, img:np.ndarray, schemas:List[Schema]) -> None:
        self.img = img
        self.schemas = schemas
    
    @classmethod
    def create_sheet_from_img(cls, img:np.ndarray, schemas:List[Schema]) -> 'RecordSheet':
        sheet = RecordSheet(img, schemas)
        mask = cls.create_line_mask(img)
        debugImgWrite(mask, type(sheet).__name__, "1mask")
        sheet.tables = cls.analyze_table_structure_as_struct(mask)
        logger.debug(f"Found {len(sheet.tables)} tables")
        for table in sheet.tables:
            logger.debug(f"table[{table.id}] = {table.rect}, {table.row_count}, {table.col_max_count}, {table.cells}")
        return sheet
        
    @classmethod
    def create_line_mask(cls, image:np.ndarray):
        """ 線画抽出用のマスク画像を作成"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # _, img_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        _, img_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        
        # 画像サイズに応じたカーネル長
        kernel_length = np.array(gray).shape[1] // 80

        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

        img_temp1 = cv2.erode(img_bin, ver_kernel, iterations=3)
        vertical_lines = cv2.dilate(img_temp1, ver_kernel, iterations=3)

        img_temp2 = cv2.erode(img_bin, hor_kernel, iterations=3)
        horizontal_lines = cv2.dilate(img_temp2, hor_kernel, iterations=3)

        mask = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        mask = cv2.dilate(mask, kernel, iterations=2)
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        return mask

    @classmethod
    def analyze_table_structure_as_struct(cls, mask:np.ndarray, min_area=500) -> List[Table]:
        """
        マスク画像から Table オブジェクトのリストを作成して返す
        """
        # 輪郭検出
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return []

        hierarchies = hierarchy[0] # [Next, Previous, First_Child, Parent]
        
        # 検索用の一時辞書 (key: contour_index, value: Tableオブジェクト)
        table_map:Dict[int, Table] = {}

        # --- Step 1: 親（表の外枠）を特定して Table オブジェクトを作成 ---
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < min_area:
                continue

            parent_idx = hierarchies[i][3]

            # 親がいない (-1) ものを「表」とみなす
            if parent_idx == -1:
                x, y, w, h = cv2.boundingRect(contour)
                # 構造体のインスタンス化
                table_obj = Table(id=i, rect=(x, y, w, h))
                table_map[i] = table_obj

        # --- Step 2: 子（セル）を特定して、親 Table に追加 ---
        bound = np.array(mask).shape[1] // 200
        logger.debug(f"bound={bound}")
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < min_area:
                continue
                
            parent_idx = hierarchies[i][3]

            # 親が存在し、その親が table_map に登録されている場合
            if parent_idx != -1 and parent_idx in table_map:
                x, y, w, h = cv2.boundingRect(contour)
                # 構造体のインスタンス化
                cell_obj = Cell(id=i, rect=(x, y, w, h))
                
                # 親のテーブルに追加
                table_map[parent_idx].add_cell(cell_obj, bound = bound)

        # --- Step 3: ソート処理 ---
        # 辞書の value (Tableオブジェクト) をリスト化
        result_tables = list(table_map.values())

        # 表を Y座標順 にソート
        result_tables.sort(key=lambda t: t.rect[1])

        return result_tables  

    def match_tables(self):
        """
        スキーマーごとに一致するTableを探す
        """ 
        for schema in self.schemas:
            for table in self.tables:
                score = schema.match(table, self.img.shape[1], self.img.shape[0])
                # logger.debug(f"Match(Schema[{schema.name}], Table[{table.id}]) = {score:.4f}")
                if score > 0.8:
                    self.matched_tables[schema.name] = table
                    logger.info(f"Schema[{schema.name}] matched Table[{table.id}]")

    def get_image(self, rect:tuple[int, int, int, int]) -> np.ndarray:
        """
        指定した領域のイメージ画像を得る
        """
        cropped_image = self.img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        return cropped_image



@dataclass_json
@dataclass
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

    FORMAT_FRAME: ClassVar[str] = "frame"
    FORMAT_TEXT: ClassVar[str] = "text"
    FORMAT_MARK: ClassVar[str] = "mark"
    SCHEMA_ALL_TEXT: ClassVar[str] = r'(.*)'
    SCHEMA_DATE_TEXT: ClassVar[str] = r'(\d*)年(\d*)月(\d*)日'

    name:str
    rect:list[int]
    parent: "Frame" = field(default=None, repr=False, metadata=config(exclude=True))
    # parent: "Frame" = config(exclude=True, metadata=field(default=None, repr=False))
    children: Dict[str, List["Frame"]] = field(default_factory=dict, init=False)
    format:str = FORMAT_FRAME
    value:str = ""
    schema:str = ""
    cluster_list_row:list['Cluster'] = field(default_factory=list, init=False)
    cluster_list_col:list['Cluster'] = field(default_factory=list, init=False)

    def __post_init__(self):
        if (self.parent is not None):
            self.parent.children[self.name] = self

    def to_json2(self):
        js = {"name": self.name, "rect": self.rect}
        if (len(self.children) > 0):
            js["children"] = list(map(lambda c: c.to_json2(), self.children.values()))
            js["cluster_list_row"] = list(map(lambda c: c.to_dict(), self.cluster_list_row))
            js["cluster_list_col"] = list(map(lambda c: c.to_dict(), self.cluster_list_col))
        return js

    """
    def __init__(self, name:str, rect:List[int], parent:"Frame"=None) -> None:
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
    """
        
    def get_children(self) -> List["Frame"]:
        self.children.values()

    def abs_rect(self) -> List[int]:
        rect = self.rect.copy()
        parent = self.parent
        while (parent is not None):
            rect[0] += parent.rect[0]
            rect[1] += parent.rect[1]
            parent = parent.parent
        return rect 
    
    def get_image(self, image):
        rect = self.abs_rect()
        cropped_image = image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        return cropped_image

    """
    def to_dict(self) -> dict:
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
    """
        
    def scan_value(self, scan):
        ptn = re.compile(self.schema)
        text = scan
        if result := ptn.search(text):
            self.value = result.group(0)
        else:
            self.value = ""
        return self.value
    
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

            cluster_list:List[Cluster] = []
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
                    new_cluster = Cluster(f"{cluster_name}{len(cluster_list)+1}", direction, child.rect[pos_index], child.rect[len_index])
                    new_cluster.frames.append(child.name)
                    bisect.insort(cluster_list, new_cluster, key=lambda c: c.pos)
            
            return cluster_list


    def create_claster_list(self):
        self.cluster_list_row = self.create_claster_list_sub(Cluster.DIRECT_ROW)
        self.cluster_list_col = self.create_claster_list_sub(Cluster.DIRECT_COL)
        return

    def get_frame_in_cluster(self, row_index, col_index) -> "Frame":
        try:
            cluster = self.cluster_list_row[row_index]
            frame_name = cluster.frames[col_index]
            frame = self.children[frame_name]
        except:
#            logger.error(f"存在しないフレームの場所にアクセスしました cluster_row={len(self.cluster_list_row)}, col={col_index}, cluster_col={len(self.cluster_list_col)}, row={row_index}")
            logger.error(f"存在しないフレームの場所にアクセスしました cluster={self}, col={col_index}, row={row_index}")
            return Frame("none", [0, 0, 0, 0])
        
        return frame

@dataclass_json
@dataclass
class Cluster:
    """
    子フレームの集合を列または行のあつまりで扱うためのクラス
    """
    # クラスターの方向
    DIRECT_COL: ClassVar[str] = 'col'  
    DIRECT_ROW: ClassVar[str] = 'row'
    NEAR_VALUE: ClassVar[int] = 5 # 1%の違い以内なら、同じ位置や長さとみなす

    name:str
    direct:str
    pos:int
    len:int
    frames:List[str] = field(default_factory=list, init=False)

    """
    def __init__(self, name, direct, pos, len) -> None:
        self.name = name
        self.direct = direct
        self.pos = pos
        self.len = len
        self.frames = []
    """

    def is_same_cluster(self, rect):
        is_same = False
        if (self.direct == Cluster.DIRECT_COL):
            is_same = ((math.isclose(self.pos, rect[0], abs_tol=Cluster.NEAR_VALUE)) and (math.isclose(self.len, rect[2], abs_tol=Cluster.NEAR_VALUE)))
        else:
            is_same = ((math.isclose(self.pos, rect[1], abs_tol=Cluster.NEAR_VALUE)) and (math.isclose(self.len, rect[3], abs_tol=Cluster.NEAR_VALUE)))
        return is_same
    
    """
    def to_dict(self):
        return {"direct": self.direct, "pos": self.pos, "len": self.len, "frames": self.frames}
    """

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



import cv2
import numpy as np

def extract_table_structure(img):

    debugImgWrite(img, "extract_table_structure", "1img")

    # 2. 二値化（白黒はっきりさせる + 色反転）
    # 背景を黒、文字・線を白にする必要があります
    thresh_value, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = cv2.bitwise_not(img_bin) # 白黒反転
    debugImgWrite(img_bin, "extract_table_structure", "2img_bin")

    # 3. カーネルの作成（ここが重要）
    # 画像の幅に基づいて、線を認識する「長さ」を決めます。
    # この値より短い線（文字の筆跡など）は無視されます。
    kernel_length = np.array(img_bin).shape[1] // 80  # 画像幅の1/40程度の長さ
    
    # 横線抽出用のカーネル (横長: [kernel_length, 1])
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # 縦線抽出用のカーネル (縦長: [1, kernel_length])
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    # 4. モルフォロジー演算で線を抽出
    # 縦線の抽出
    img_temp1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(img_temp1, ver_kernel, iterations=3)
    
    # 横線の抽出
    img_temp2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(img_temp2, hor_kernel, iterations=3)

    # 5. 縦線と横線を結合（+ 重み付けで線を太くして結合部を補完）
    # これで「文字のない、枠線だけの画像」が完成します
    table_mask = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    
    # 結合部が途切れている場合のために少し膨張させる
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    table_mask = cv2.dilate(table_mask, kernel, iterations=2)
    
    # 二値化しなおしてクリーンにする
    _, table_mask = cv2.threshold(table_mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    debugImgWrite(table_mask, "extract_table_structure", "3table_mask")

    return table_mask

def extract_table_structure2(table_mask):
    # 6. 輪郭検出 (RETR_TREEで階層構造を取得)
    contours, hierarchy = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 結果描画用のカラー画像
    result_img = cv2.cvtColor(table_mask, cv2.COLOR_GRAY2BGR)

    tables = [] # 表の外枠
    cells = []  # セル

    for i, contour in enumerate(contours):
        # 面積で小さすぎるゴミを除去
        area = cv2.contourArea(contour)
        if area < 500: 
            continue

        # 矩形情報を取得 (x, y, width, height)
        x, y, w, h = cv2.boundingRect(contour)
        
        # 階層情報の確認: hierarchy[0][i] = [Next, Previous, First_Child, Parent]
        # Parentが -1 なら、それは一番外側の枠（表の外郭）である可能性が高い
        parent_idx = hierarchy[0][i][3]
        
        if parent_idx == -1:
            # 親がいない＝表全体の外枠
            tables.append((x, y, w, h))
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 3) # 赤枠
        else:
            # 親がいる＝セル
            cells.append((x, y, w, h))
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2) # 緑枠

    debugImgWrite(result_img, "extract_table_structure", "4result_img")

    return tables, cells

# 実行
# result_image, table_rects, cell_rects = extract_table_structure('scan_image.png')
# cv2.imwrite('result.png', result_image)




class FrameDetector:
    def __init__(self) -> None:
        self.image = None
        self.contours = None
        self.hierarchy = None
    
    def parse_image(self, image):
        self.img_debug = image.copy()
        debugImgWrite(self.img_debug, type(self).__name__, "1input")

        # 画像をグレースケールにして白黒反転する
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, self.image = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        logger.debug(f"auto threshold={ret}")
        debugImgWrite(self.image, type(self).__name__, "1gray")

        # 縦と横成分だけを抽出して、表の枠だけを抽出する
        result_image = extract_table_structure(img_gray)
        debugImgWrite(result_image, type(self).__name__, "1result")
        self.image = result_image

        # 入力された領域をRootフレームとする
        img_w = self.image.shape[1]
        img_h = self.image.shape[0]
        self.root_frame = Frame("root", [0, 0, img_w, img_h], None)
        self.cont_index_tbl = {}

        # 輪郭を抽出する
        self.contours, self.hierarchy = cv2.findContours(self.image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return self.root_frame        

    def get_root_frame(self) -> Frame:
        return self.root_frame
    
    def get_image(self, frame:Frame):
        rect = frame.abs_rect()
        frame_img = self.image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        return(frame_img)

    def get_debug_image(self, frame:Frame):
        rect = frame.abs_rect()
        frame_img = self.img_debug[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        return(frame_img)

    def detect_sub_frames(self, target_frame:Frame, level) -> List[Frame]:
        if (target_frame == self.root_frame):
            child_index = 0
        else:
            my_index = self.cont_index_tbl[target_frame.name]
            child_index = self.hierarchy[0][my_index][2]   # 最初の子供の輪郭

        while (child_index != -1):
            #get_frames_sub(self.image, self.contours, self.hierarchy, cont_index, 0, target_frame)
            trim_offset = 10
            color = [(0, 255, 0), (128, 128, 0), (0, 0, 255)]
            skip_color = (0, 128, 128)

            # 自身の領域情報を得る
            contour = self.contours[child_index]
            x,y,w,h = cv2.boundingRect(contour)
            img_w = self.image.shape[1]
            img_h = self.image.shape[0]
            w_rel = w / img_w * 100
            h_rel = h / img_h * 100

            # 矩形であることを確認する
            approx = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)

            # ある程度の大きさがある矩形領域だけをフレーム作成の対象にする
            if ((w_rel > 8) and (h_rel > 2) and (w > 10) and (h > 5) and (len(approx) == 4)):
                # Frameクラスを作る
                frame = Frame(name=f"frame_{child_index}", rect=[x - target_frame.rect[0], y - target_frame.rect[1], w, h], parent=target_frame)
                self.cont_index_tbl[frame.name] = child_index
                logger.info(f"create child frame: {frame}")
                cv2.rectangle(self.img_debug, (x, y), (x+w, y+h), color[level], trim_offset)
                cv2.putText(self.img_debug, f"frame: {frame.name}", (x, y-20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0))
            else:
                # 小さい領域は無視する
                logger.debug(f"skipped small area {x},{y},{w_rel}%,{h_rel}%  approx={len(approx)}")
                cv2.rectangle(self.img_debug, (x, y), (x+w, y+h), skip_color, trim_offset)

            child_index = self.hierarchy[0][child_index][0]   # 次の輪郭

        target_frame.create_claster_list()
        debugImgWrite(self.img_debug, type(self).__name__, "2output")

        return target_frame.get_children()

    def detect_table_frame(self, target_frame:Frame, level) -> List[Frame]:
        """
        表イメージを解析する
        @param img:画像イメージ(カラー)
        """
        img = self.get_image(target_frame)
        img_debug = self.get_debug_image(target_frame)
        img_w = img.shape[1]
        img_h = img.shape[0]

        # 罫線検出
        lines = cv2.HoughLinesP(img, rho=1, theta=np.pi/2, threshold=80, minLineLength=img_w/2, maxLineGap=10)
        cols = []
        rows = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if ((abs(x1-x2) < 10) and (abs(y1-y2) > 10)):
                # 縦線
                cols.append(x1)
                cv2.line(img_debug, (x1,y1), (x2,y2), (255,0,0), 3)
            elif ((abs(x1-x2) > 10) and (abs(y1-y2) < 10)):
                # 横線
                rows.append(y1)
                cv2.line(img_debug, (x1,y1), (x2,y2), (0,255,0), 3)
            else:
                print(f"処理できない線がみつかりました ({x1}, {y1}-({x2},{y2}))")
                cv2.line(img_debug, (x1,y1), (x2,y2), (0,0,255), 3)

        # 列と行のセルを求める
        col_spans = trimPosList(sorted(cols), skip=4)
        row_spans = trimPosList(sorted(rows), skip=4)
        debugImgWrite(img_debug, "step4", "table")

        for row in range(min(40, len(row_spans))):
            (row_start, row_end) = row_spans[row]
            for col in range(min(10, len(col_spans))):
                (col_start, col_end) = col_spans[col]
                cell = Frame(f"{target_frame.name}_{row}{col}", [col_start, row_start, col_end-col_start, row_end-row_start], target_frame)
                #print(f"create child frame: {cell.__dict__}")
        target_frame.create_claster_list()

        return target_frame.get_children()


def trimPosList(pos_list:List[int], skip:int):
    span_list = []
    start = pos_list[0]
    for pos in pos_list:
        if (start + skip < pos):
            span = (start+1, pos-1)
            span_list.append(span)
        start = pos
    
    return span_list



#import frame.trim_report_frame as trim_report_frame
from .text_scan import VisonImgTextReader
def scan_frame(frame:Frame, scanner:VisonImgTextReader):
    if (len(frame.children) > 0):
        for child in frame.children.values():
            scan_frame(child, scanner)
    else:
        rect = frame.abs_rect()
        text = scanner.extract_text_from_region(rect[0], rect[1], rect[2], rect[3])
        frame.value = text
        #print(f"scanned text:{text} from {frame.name}:{rect}")

# main
if __name__ == '__main__':
    test_file = "./record/202403/202403B01.JPG"
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

    # 取得したフレーム内の文字を読み込む
    scan_frame(root, img_reader)

    # 取り出した情報を表示する
    print(f"head_date: {head_date.value}")
    print(f"head_member: {head_member.value}")
    print(f"head_wed: {head_wed.value}")
    
    print(f"course_route: {course_route.value}")
    print(f"course_page: {course_page.value}")

    # メイン部分
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

    t1 = Frame("test", [0, 0, 100, 100], None)
    t2 = Frame("child", [10, 10, 50, 50], t1)
    t31 = Frame("sub-child31", [10, 10, 50, 50], t2)
    t32 = Frame("sub-child32", [50, 10, 50, 50], t2)
    t1j = t1.to_json2()
    print(f"t1j={t1j}")
    with open("./tmp/test_info.json", "w") as f:
        json.dump(t1j, f, ensure_ascii=False, indent=4)

    # フレーム情報をJSONデータファイルとして書き込む
    d = root.to_json2()
    print(f"d={d}")
    with open("./tmp/frame_info.json", "w") as f:
        f.write(str(d))
        json.dump(d, f, ensure_ascii=False, indent=2)

    exit()



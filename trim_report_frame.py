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

   
@dataclass_json
@dataclass
class CellInfo:
    # セルの形式
    FORMAT_TEXT: ClassVar[str] = 'text'   # テキスト形式のセル
    FORMAT_MARK: ClassVar[str] = 'mark'   # マル印形式のセル
    FORMAT_TABLE: ClassVar[str] = 'table'  # 子供のセルを囲んでいるセル

    cell_index: int
    cell_rect: tuple[int, int, int, int]
    format: str = FORMAT_TEXT
    col_cluster_id: int = 0
    row_cluster_id: int = 0

    NEAR_VALUE: ClassVar[int] = 10   # 1%の違い以内なら、同じ位置や長さとみなす
    def __eq__(self, other):
        return math.isclose(self.rect[0], other.rect[0], abs_tol=CellInfo.NEAR_VALUE) and math.isclose(self.rect[2], other.rect[2], abs_tol=CellInfo.NEAR_VALUE)

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __lt__(self, other):
        if (self.rect[0] - other.rect[0] < -CellInfo.NEAR_VALUE) : return True
        if (math.isclose(self.rect[0], other.rect[0], abs_tol=CellInfo.NEAR_VALUE) and (self.rect[2] - other.rect[2] < -CellInfo.NEAR_VALUE)): return True
        return False

@dataclass_json
@dataclass
class ClusterInfo:
    # クラスターの方向
    DIRECT_COL: ClassVar[str] = 'direct_col'  
    DIRECT_ROW: ClassVar[str] = 'direct_row'
    NEAR_VALUE: ClassVar[int] = 10 # 1%の違い以内なら、同じ位置や長さとみなす

    direction: str
    pos: int
    len: int
    cells: list[int] = field(default_factory=list)
    index: int = 0

    def is_same_cluster(self, rect):
        
        is_same = False
        if (self.direction == ClusterInfo.DIRECT_COL):
            is_same = ((math.isclose(self.pos, rect[0], abs_tol=ClusterInfo.NEAR_VALUE)) and (math.isclose(self.len, rect[2], abs_tol=ClusterInfo.NEAR_VALUE)))
        else:
            is_same = ((math.isclose(self.pos, rect[1], abs_tol=ClusterInfo.NEAR_VALUE)) and (math.isclose(self.len, rect[3], abs_tol=ClusterInfo.NEAR_VALUE)))
        return is_same

@dataclass_json
@dataclass
class FrameInfo:
    # フレームの情報。フレームとは、複数のセルと縦と横のクラスター情報から構成される。
    rect: tuple[int, int, int, int]
    cells: list[CellInfo] = field(default_factory=list)
    col_cluster_list: list[ClusterInfo] = field(default_factory=list)
    row_cluster_list: list[ClusterInfo] = field(default_factory=list)



            




def to_dict(obj):
    if isinstance(obj, dict):
        return obj
    elif isinstance(obj, list):
        return [to_dict(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(to_dict(x) for x in obj)
    else:
        return asdict(obj)
        

class MyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, CellInfo):
      return obj.dump_json()
    return super().default(obj)



def trim_paper_frame(img):
    """
    黒地に映ったA4用紙の輪郭を検出する
    @param img:image 暗い背景に写された白い記録用紙の写真
    @return :image 記録用紙をトリミングした画像イメージ
      記録用紙をスキャナーで読み取った場合は暗い背景がないので、元の画像イメージをそのまま返す
    """
    img_debug = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 画像処理のノイズ除去
    ret, img_gray2 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    util.debugImgWrite(img_debug, inspect.currentframe().f_code.co_name, "1input")            

    # 境界線を見つける
    contours, hierarchy = cv2.findContours(
        img_gray2, 
        cv2.RETR_EXTERNAL,         # 一番外側の輪郭のみ 
        cv2.CHAIN_APPROX_SIMPLE   # 輪郭座標の省略
        ) 
    for i, contour in enumerate(contours):
        # 傾いていない外接する矩形領域を求める
        x,y,w,h = cv2.boundingRect(contour)
        
        # 画像の50%以上の大きさを占める輪郭をA4用紙の輪郭と扱う
        # print(img_gray.shape)
        if ((w > img_gray2.shape[1] * 0.5) and (h > img_gray2.shape[0] * 0.5)):
            # 領域の近似形状を求める
            epsilon = 0.05*cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if (len(approx) < 4):
                continue
            (top_left, top_right, bot_left, bot_right) = util.calcRectEdge(contour)
            trim_img = util.trimAsRectangle(img, top_left, top_right, bot_left, bot_right)
            if (__debug__):
                cv2.drawContours(img_debug, contours, i, (255, 0, 0), 2)
                cv2.rectangle(img_debug,(x,y),(x+w-1,y+h-1),(0,255,0),2)
                cv2.circle(img_debug, top_left, 20, (0, 255, 255), -1)
                cv2.circle(img_debug, top_right, 20, (0, 255, 255), -1)
                cv2.circle(img_debug, bot_left, 20, (0, 255, 255), -1)
                cv2.circle(img_debug, bot_right, 20, (0, 255, 255), -1)
                util.debugImgWrite(img_debug, inspect.currentframe().f_code.co_name, "2output")            
            return (trim_img)
    
    #　見つからなかった場合は、用紙全体を写したか、白地の背景と解釈して、イメージ領域をそのまま返す
    print("Paper frame not found. Returned whole picture as a paper.")
    return(img)

def trim_inner_mark(img):
    """
    四隅の位置合わせマークで囲まれた区画を抽出し、台形補正する
    位置合わせマークは、３つの三角形と１つの正方形(右下)
    @param img: 元の画像(白地が背景)
    @return: 抽出された画像 
    """
    # 白地のA4用紙のレポートから、▲と■マークを検出する
    img_debug = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 画像処理のノイズ除去
    ret, img_gray2 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    util.debugImgWrite(img_gray2, inspect.currentframe().f_code.co_name, "1gray")
  
    # 境界線を見つける
    contours, hierarchy = cv2.findContours(
        img_gray2, 
        cv2.RETR_EXTERNAL,         # 2段階の輪郭で取得。マークには子供がいないはず 
        cv2.CHAIN_APPROX_SIMPLE
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
        if (area < min_mark_w * min_mark_h * 0.5):
            # print(f" detect edge mark: skiped small area size={area}")
            continue
        elif (area > max_mark_w * max_mark_h):
            # print(f" detect edge mark: skiped big area size={area}")
            continue
            
        # 領域の形状を見つける
        epsilon = 0.02*cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if (len(approx) == 3): 
            # 塗りつぶした三角形
            cv2.drawContours(img_debug, [approx], 0, (0, 255, 255), 5)
            cv2.putText(img_debug, f"size={area}", approx[0][0], cv2.FONT_HERSHEY_DUPLEX, 0.8, (0))
            triangles += 1
            tri_area.append(approx)
        elif (len(approx) == 4):
            # 塗りつぶした４角形かを確認
            cv2.drawContours(img_debug, [approx], 0, (0, 255, 0), 5)
            cv2.putText(img_debug, f"size={area}", approx[0][0], cv2.FONT_HERSHEY_DUPLEX, 0.8, (0))
            box += 1
            box_area.append(approx)
        else:
            cv2.drawContours(img_debug, [approx], 0, (0, 0, 255), 5)
            cv2.putText(img_debug, f"size={area}, edge={len(approx)}", approx[0][0], cv2.FONT_HERSHEY_DUPLEX, 0.8, (0))



    if ((len(tri_area) == 3) and (len(box_area) == 1)):
        # 四隅を発見できた
        center = (img.shape[1]/2, img.shape[0]/2)
        right_bot = min(box_area[0], key=lambda x: np.linalg.norm(x[0]-center))[0] # 四角形は右下固定
        tri = []
        for t in tri_area:
            tri.append(min(t, key=lambda x: np.linalg.norm(x[0]-center))[0])

        # 三つの三角について、右下からの距離でソートして、遠い順に左上、右上、左下を求める。
        (left_bot, right_top, left_top) = sorted(tri, key=lambda x: np.linalg.norm(x-right_bot))
        new_img = util.trimAsRectangle(img, left_top, right_top, left_bot, right_bot)

        cv2.putText(img_debug, "left_top", left_top+(0,-20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0))
        cv2.putText(img_debug, "right_top", right_top+(0,-20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0))
        cv2.putText(img_debug, "left_bot", left_bot+(0,20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0))
        cv2.putText(img_debug, "right_bot", right_bot+(0,20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0))
        points = np.array([left_top, right_top, right_bot, left_bot]).reshape(1, -1, 2)
        cv2.polylines(img_debug, points, isClosed=True, color=(255, 0, 0), thickness=5)  
        util.debugImgWrite(img_debug, inspect.currentframe().f_code.co_name, "2box")
    else:
        # 四隅が見つからない場合は、イメージをそのまま返す
        print(f"四隅のマークを検出できませんでした。元のイメージをそのまま返します。(三角形:{len(tri_area)}, 四角形:{len(box_area)})")
        new_img = img
        
    util.debugImgWrite(img_debug, inspect.currentframe().f_code.co_name, "2box")
    return(new_img)

def trim_inner_mark2(img):
    """
    四隅の位置合わせマークで囲まれた区画を抽出し、台形補正する
    位置合わせマークは、３つの中空の正方形と１つの塗りつぶされた正方形(右下)
    @param img: 元の画像(白地が背景)
    @return: 抽出された画像 
    """
    # 白地の画像なので、反転させて処理する
    img_debug = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 画像処理のノイズ除去
    ret, img_inv = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    util.debugImgWrite(img_inv, inspect.currentframe().f_code.co_name, "1gray")
  
    # 境界線を見つける
    contours, hierarchy = cv2.findContours(
        img_inv, 
        cv2.RETR_EXTERNAL,         # 外輪だけを抽出する
        cv2.CHAIN_APPROX_SIMPLE
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
        #print(f"detect area = {rect}")
        if ((width < min_mark_w) or (height < min_mark_h)):
            #print(f" detect edge mark: skiped small area rect={rect}")
            continue
        elif ((width > max_mark_w) or (height > max_mark_h)):
            #print(f" detect edge mark: skiped big area size={rect}")
            continue
            
        # 領域の形状をチェックする
        #print(f" check edge mark: rect={rect}")        
        epsilon = 0.02*cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if (len(approx) == 4):
            # マークの形状をチェックする
            x, y, w, h = cv2.boundingRect(approx)
            cropped_image = img_inv[y:y+h, x:x+w]
            sub_contours, hierarchy2 = cv2.findContours(
                cropped_image, 
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
            )
            if (len(sub_contours) == 1):
                # 塗りつぶされたマーク(右下)
                anker += 1
                anker_area.append(approx)
                cv2.drawContours(img_debug, [approx], 0, (255, 0, 0), 5)
                cv2.putText(img_debug, f"rect={rect}", approx[0][0], cv2.FONT_HERSHEY_DUPLEX, 0.8, (0))
            elif (len(sub_contours) == 2):
                # 真ん中が空いたマーク(左上、右上、左下のどれか)
                box += 1
                box_area.append(approx)
                cv2.drawContours(img_debug, [approx], 0, (0, 255, 0), 5)
                cv2.putText(img_debug, f"rect={rect}", approx[0][0], cv2.FONT_HERSHEY_DUPLEX, 0.8, (0))
            else:
                cv2.drawContours(img_debug, [approx], 0, (0, 0, 128), 5)
                cv2.putText(img_debug, f"rect={rect}, edge={len(approx)}", approx[0][0], cv2.FONT_HERSHEY_DUPLEX, 0.8, (0))

        else:
            cv2.drawContours(img_debug, [approx], 0, (0, 0, 255), 5)
            cv2.putText(img_debug, f"rect={rect}, edge={len(approx)}", approx[0][0], cv2.FONT_HERSHEY_DUPLEX, 0.8, (0))


    if ((len(anker_area) == 1) and (len(box_area) == 3)):
        # 四隅を発見できた
        found = True
        center = (img.shape[1]/2, img.shape[0]/2)
        right_bot = min(anker_area[0], key=lambda x: np.linalg.norm(x[0]-center))[0] # 四角形は右下固定
        tri = []
        for t in box_area:
            tri.append(min(t, key=lambda x: np.linalg.norm(x[0]-center))[0])

        # 三つのマーカーについて、右下からの距離でソートして、遠い順に左上、右上、左下を求める。
        (left_bot, right_top, left_top) = sorted(tri, key=lambda x: np.linalg.norm(x-right_bot))

        # 台形補正する
        new_img = util.trimAsRectangle(img, left_top, right_top, left_bot, right_bot)

        cv2.putText(img_debug, "left_top", left_top+(0,-20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0))
        cv2.putText(img_debug, "right_top", right_top+(0,-20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0))
        cv2.putText(img_debug, "left_bot", left_bot+(0,20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0))
        cv2.putText(img_debug, "right_bot", right_bot+(0,20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0))
        points = np.array([left_top, right_top, right_bot, left_bot]).reshape(1, -1, 2)
        cv2.polylines(img_debug, points, isClosed=True, color=(255, 255, 0), thickness=5)  
        util.debugImgWrite(img_debug, inspect.currentframe().f_code.co_name, "2box")
    else:
        # 四隅が見つからない場合は、イメージをそのまま返す
        found = False
        print(f"四隅のマークを検出できませんでした。元のイメージをそのまま返します。(box:{len(box_area)}, anker:{len(anker_area)})")
        new_img = img
        
    util.debugImgWrite(img_debug, inspect.currentframe().f_code.co_name, "2box")
    return(new_img, found)


def getDescAreaInfo2(img, inv=False) -> FrameInfo:
    """
    記録用のテンプレート画像から、区画情報のスキーマーを計算する
    @param img: レポートの画像イメージ(白地)
    @param inv:bool 画像を反転させる (True:黒地に白の画像の場合、False:白地に黒の画像の場合)
    @return 区画情報 
    """
    # 画像処理用のイメージを作る
    img_debug = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_gray2 = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    util.debugImgWrite(img_gray2, inspect.currentframe().f_code.co_name, "1input")

    cell_list = getFrameInfo(img_gray2, img_debug)

    col_cluster_list = []
    for cell in cell_list:
        found = False
        for col_cluster in col_cluster_list:
            if col_cluster.is_same_cluster(cell.cell_rect):
                col_cluster.cells.append(cell.cell_index)
                found = True
                break
        if (not found):
            new_cluster = ClusterInfo(ClusterInfo.DIRECT_COL, cell.cell_rect[0], cell.cell_rect[2])
            new_cluster.index = len(col_cluster_list)+1
            new_cluster.cells.append(cell.cell_index)
            col_cluster_list.append(new_cluster)
    
    row_cluster_list: List[ClusterInfo] = []
    for cell in cell_list:
        found = False
        for row_cluster in row_cluster_list:
            if row_cluster.is_same_cluster(cell.cell_rect):
                row_cluster.cells.append(cell.cell_index)
                found = True
                break
        if (not found):
            new_cluster = ClusterInfo(ClusterInfo.DIRECT_ROW, cell.cell_rect[1], cell.cell_rect[3])
            new_cluster.index = len(row_cluster_list)+1
            new_cluster.cells.append(cell.cell_index)
            row_cluster_list.append(new_cluster)
  
    
    util.debugImgWrite(img_debug, inspect.currentframe().f_code.co_name, "2output")
    frame = FrameInfo(rect = (0, 0, img.shape[1], img.shape[0]))
    frame.cells = cell_list
    frame.col_cluster_list = col_cluster_list
    frame.row_cluster_list = row_cluster_list

    #print(f"frame: {frame}")

    return frame

from typing import List
def getFrameInfo(img_gray2, img_debug) -> List[CellInfo]:
    img_w = img_gray2.shape[1]
    img_h = img_gray2.shape[0]

    # 外側の輪郭を見つける
    contours, hierarchy = cv2.findContours(
        img_gray2, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
        ) 
    
    # 第一階層の輪郭を追って情報を得る
    trim_offset = 10
    info: list[CellInfo] = []
    cont_index = 0
    while (cont_index != -1):
        trim_offset = 10
        color = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]

        # 領域情報を得る
        contour = contours[cont_index]
        area = cv2.contourArea(contour)
        x,y,w,h = cv2.boundingRect(contour)
        img_w = img_gray2.shape[1]
        img_h = img_gray2.shape[0]
        x_rel = int(x * 1000 / img_w)
        y_rel = int(y * 1000 / img_h)
        w_rel = int(w * 1000 / img_w)
        h_rel = int(h * 1000 / img_h)
        
        # 大きな領域だけを処理する
        if ((w_rel > 10) and (h_rel > 10)):
            # 領域情報を作成する
            cell_info = CellInfo(cont_index, (x_rel, y_rel, w_rel, h_rel))
            #print(f"cell_info:{cell_info}")
            #j = cell_info.to_json()
            #print(f"json:{j}")
            info.append(cell_info)

            cv2.rectangle(img_debug, (x, y), (x+w, y+h), (0, 255, 0), trim_offset)
            cv2.putText(img_debug, f"info: {cell_info}", (x, y-20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0))
        else:
            print(f"skipped small area {x},{y},{w},{h}")

        cont_index = hierarchy[0][cont_index][0]   # 次の領域

    # print(f"info:{info}")
    # j = CellInfo.schema().dumps(info, many=True)
    # print(f"json_info:{j}")
    return (info)
    
    





def getDescAreaInfo(img, inv=False):
    """
    記録用のテンプレート画像から、区画情報のスキーマーを計算する
    @param img: レポートの画像イメージ(白地)
    @param inv:bool 画像を反転させる (True:黒地に白の画像の場合、False:白地に黒の画像の場合)
    @return 区画情報のスキーマー(JSON形式) 
    """
    # 画像処理用のイメージを作る
    img_debug = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_gray2 = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_w = img_gray2.shape[1]
    img_h = img_gray2.shape[0]
    util.debugImgWrite(img_gray2, inspect.currentframe().f_code.co_name, "1input")

    # 境界線を見つける
    contours, hierarchy = cv2.findContours(
        img_gray2, 
        cv2.RETR_TREE, 
        cv2.CHAIN_APPROX_SIMPLE
        ) 
    
    # 第一階層の輪郭を追って情報を得る
    trim_offset = 10
    info = []
    cont_index = 0
    while (cont_index != -1):
        area_info = get_rectangle_info(img_gray2, contours, hierarchy, cont_index, 0, img_debug)
        if area_info is not None:
            info.append(area_info)
        cont_index = hierarchy[0][cont_index][0]   # 次の領域

    util.debugImgWrite(img_debug, inspect.currentframe().f_code.co_name, "2output")
    return (info)

def get_rectangle_info(image, contours, hierarchy, target_cont_index, level, img_debug):
    """
    矩形領域の情報をJSON形式で返す
    Args:
        image: 入力画像
        contours: 輪郭のリスト
        hierarchy: 輪郭の階層情報
        target_cont_index: 対象の輪郭のインデックス
        level: 階層レベル

    Returns:
        矩形領域の情報 (JSON形式)
    """

    trim_offset = 10
    color = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]

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
    if ((w_rel < 0.01) or (h_rel < 0.01)):
        print(f"skipped small area {x},{y},{w},{h}")
        return None

    # 子輪郭の情報リストを得る
    child_cont_index = hierarchy[0][target_cont_index][2]
    child_info_list = []
    while (child_cont_index != -1):
        child_info = get_rectangle_info(image, contours, hierarchy, child_cont_index, level + 1, img_debug)
        if child_info is not None:
            child_info_list.append(child_info)
        child_cont_index = hierarchy[0][child_cont_index][0]   # 次の領域

    detect_img = image[max(y-trim_offset, 0):min(y+h+trim_offset, img_h), max(x-trim_offset, 0):min(x+w+trim_offset, img_w)]

    # JSON形式で領域情報を作成する
    info = {
        "name": f"name{target_cont_index}",
        "area": {"x:": x_rel, "y": y_rel, "width": w_rel, "height": h_rel }, 
        "type": "text", 
        "children": child_info_list
    }
    
    cv2.rectangle(img_debug, (x, y), (x+w, y+h), color[level], trim_offset)
    cv2.putText(img_debug, f"info: {info}", (x, y-20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0))
    print(f"area={info}")
    
    return info


def getDescArea(img, inv=False):
    """
    レポート画像から、記述画像の部分を取り出す
    @param img: レポートの画像イメージ(白地)
    @param inv:bool 
    """
    # 画像処理用のイメージを作る
    img_debug = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_gray2 = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_w = img_gray2.shape[1]
    img_h = img_gray2.shape[0]
    util.debugImgWrite(img_gray2, inspect.currentframe().f_code.co_name, "1input")

    # 境界線を見つける
    contours, hierarchy = cv2.findContours(
        img_gray2, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
        ) 

    # 一番外側の枠を分類する
    main = None
    main_pos = (0, 0)
    head = None
    head_pos = (0, 0)
    place = None
    place_pos = (0, 0)
    unknown_area = []
    trim_offset = 10
    for i, contour in enumerate(contours):
        # 小さな領域はマークとみなして、スキップする
        area = cv2.contourArea(contour)
        # print(f"area = {area}")
        if (area < img_w * img_h / 500):
            continue
            
        # 領域の位置や大きさで分類する
        x,y,w,h = cv2.boundingRect(contour)
        detect_img = img[max(y-trim_offset, 0):min(y+h+trim_offset, img_h), max(x-trim_offset, 0):min(x+w+trim_offset, img_w)]
        #pt1 = (max(x-trim_offset, 0), max(y-trim_offset, 0))
        #pt2 = (min(x+w+trim_offset, img_w), )
        if ((w > img_w * 0.5) and (h > img_h * 0.6)):
            # メイン記述領域
            main = detect_img
            main_pos = (x, y)
            cv2.rectangle(img_debug, (x, y), (x+w, y+h), (0, 255, 0), trim_offset)
            cv2.putText(img_debug, f"main: ({x},{y}) {w}x{h}", (x, y-20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0))
        elif((x < img_w * 0.2) and (y < img_h * 0.1) and (w > img_w * 0.3) and (h < img_h * 0.1)):
            # ヘッダ記述領域
            # head = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            head = detect_img
            head_pos = (x, y)
            cv2.rectangle(img_debug,  (x, y), (x+w, y+h), (0, 0, 255), trim_offset)
            cv2.putText(img_debug, f"head: ({x},{y}) {w}x{h}", (x, y-20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0))
        elif((x > img_w * 0.6) and (y < img_h * 0.1) and (w > img_w * 0.1) and (h < img_h * 0.1)):
            # 場所記述領域
            place = detect_img
            place_pos = (x, y)
            cv2.rectangle(img_debug,  (x, y), (x+w, y+h), (0, 255, 255), trim_offset)
            cv2.putText(img_debug, f"place: ({x},{y}) {w}x{h}", (x, y-20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0))
        else:
            # 不明な場所
            print(f"分類できない領域が見つかりました  ({x},{y}) {w}x{h}")
            unknown_area.append(contour)
            cv2.drawContours(img_debug, contours, i, (255, 0, 0), 2)
            cv2.putText(img_debug, f"unknown: ({x},{y}) {w}x{h}", (x, y-20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0))
    
        util.debugImgWrite(img_debug, inspect.currentframe().f_code.co_name, "2output")
    return (main, main_pos, head, head_pos, place, place_pos)



# main
g_skipText=False
if __name__ == '__main__':
    args = sys.argv
    if 2 > len(args):
        print(f"Usage {args[0]} [--skipText] image_file")
    else:
        util.debugTmpImgRemove()
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
                    img = cv2.imread(file)
                    trim_img = trim_paper_frame(img)
                    trim_img2 = trim_inner_mark2(trim_img)
                    info = getDescAreaInfo2(trim_img2)
                    dict_info = asdict(info)
                    print(f"schema info: {dict_info}")

                    json_data =  FrameInfo.schema().dumps(info, indent=4)
                    #json_data = json.dumps(dict_info)
                    print(f"json_data = {json_data}")

                    # JSONデータをファイルに書き込み
                    with open("./tmp/area_info.json", "w") as f:
                        json.dump(json_data, f, indent=4)
                    main, main_pos, head, head_pos, place, place_pos  = getDescArea(trim_img2)



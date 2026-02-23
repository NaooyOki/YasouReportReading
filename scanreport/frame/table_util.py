import cv2
import numpy as np
import json
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Optional
import logging

from ..util import *

# ログ
logger = logging.getLogger(__name__)


# --- 前回の定義（参照用） ---
@dataclass
class Cell:
    """セル情報を格納するデータクラス"""
    id: int                         # 輪郭のインデックス
    rect: Tuple[int, int, int, int] # (x, y, width, height)
    
    @property
    def area(self):
        """面積を返すプロパティ（必要に応じて）"""
        return self.rect[2] * self.rect[3]

@dataclass
class Table:
    """表情報を格納するデータクラス"""
    id: int                         # 輪郭のインデックス
    rect: Tuple[int, int, int, int] # (x, y, width, height)
    cells: List[List[Cell]] = field(default_factory=list) # 子要素（セル）のリスト  行の配列に、列の配列を加えた2次元配列

    def add_cell(self, cell: Cell, bound: int = 5):
        # 一致する行を探す
        target_row = None
        for row in self.cells:
            if (abs(row[0].rect[1] - cell.rect[1]) < bound):
                target_row = row
                break
            elif (row[0].rect[1] > cell.rect[1]):
                target_row = []
                self.cells.insert(self.cells.index(row), target_row)
                break
        if (target_row is None):
            target_row = []
            self.cells.append(target_row)
        
        # 列を挿入し、ソートする
        target_row.append(cell)
        target_row.sort(key=lambda c: c.rect[0])
    
    @property
    def row_count(self) -> int:
        """行数を返す"""
        return len(self.cells)
    
    @property
    def col_max_count(self) -> int:
        """列数を返す"""
        max_col = 0
        for row in self.cells:
            max_col = max(max_col, len(row))
        return max_col

    @property
    def col_count(self, row:int) -> int:
        """列数を返す"""
        try:
            return len(self.cells[row])
        except IndexError:
            return 0

    def get_cell(self, row: int, col: int) -> Optional[Cell]:
        """指定した行列のセルを返す"""
        try:
            return self.cells[row][col]
        except IndexError:
            logger.error(f"TableSchma.get_cell() invalid index: row={row}, col={col}")
            return None

        
            





# --- 新しい定義: TableSchema ---

@dataclass
class CellSchema:
    """
    セルのテンプレート情報を管理するクラス。
    セルに記述される情報の型を持つ。
    """
    name: str       # セルの名前
    datatype: str   # セルの中に記述される情報の型
    format: str     # セルの中に記述される情報のフォーマット

    def to_json(self) -> str:
        """JSON文字列にシリアライズ"""
        return json.dumps(asdict(self), ensure_ascii=True)

    @classmethod
    def from_json(cls, json: dict) -> 'CellSchema':
        """JSON文字列からデシリアライズ"""
        return cls(**json)
    


@dataclass
class TableSchema:
    """
    表のテンプレート情報を管理するクラス。
    画像解像度に依存しない相対座標と、期待される構造を持つ。
    """
    name: str  # "請求金額欄", "明細表" などの識別名
    relative_rect: Tuple[float, float, float, float]  # (x, y, w, h) ※0.0〜1.0の割合
    cells: List[List[CellSchema]] = field(default_factory=list)
    
    # 許容誤差（マッチング判定用）
    tolerance: float = 0.05 

    def to_json(self) -> str:
        """JSON文字列にシリアライズ"""
        return json.dumps(asdict(self), ensure_ascii=True)

    @classmethod
    def from_json(cls, json: dict) -> 'TableSchema':
        """JSON文字列からデシリアライズ"""
        return cls(**json)

    def add_cells(self, cells: List[List[CellSchema]]):
        self.cells = cells
    
    @property
    def get_cell(self, row: int, col: int) -> Optional[CellSchema]:
        """指定した行列のセルを返す"""
        try:
            return self.cells[row][col]
        except IndexError:
            logger.error(f"TableSchma.get_cell() invalid index: row={row}, col={col}")
            return None

    @property
    def expected_rows(self) -> int:
        """行数を返す"""
        return len(self.cells)
    
    @property
    def expected_cols(self, row:int) -> int:
        """列数を返す"""
        try:
            return len(self.cells[row])
        except IndexError:
            return 0

    @property
    def expected_max_cols(self) -> int:
        """最大の列数を返す"""
        max_col = 0
        for row in self.cells:
            max_col = max(max_col, len(row))
        return max_col


    def add_cells_from_table(self, table: Table):
        self.cells = []
        for r, row in enumerate(table.cells):
            schema_row = []
            for c, cell in enumerate(row):
                # CellオブジェクトからデフォルトのCellSchemaを作成
                schema_row.append(CellSchema(
                    name=f"cell_{r}_{c}",
                    datatype="text",
                    format=""
                ))
            self.cells.append(schema_row)


    def match(self, table: Table, image_width: int, image_height: int) -> float:
        """
        検出されたTableが、このSchemaとどれくらい一致しているかをスコア(0.0-1.0)で返す。
        
        Args:
            table: 検出されたTableオブジェクト
            image_width: 画像全体の幅
            image_height: 画像全体の高さ
        Returns:
            score: 一致度（1.0に近いほど一致）
        """
        # 1. 検出された表の相対座標を計算
        tx, ty, tw, th = table.rect
        rel_x = tx / image_width
        rel_y = ty / image_height
        rel_w = tw / image_width
        rel_h = th / image_height

        # 2. 座標の差分（距離）を計算
        sx, sy, sw, sh = self.relative_rect
        
        # 中心点のズレ
        center_dist = ((rel_x + rel_w/2) - (sx + sw/2))**2 + ((rel_y + rel_h/2) - (sy + sh/2))**2
        # サイズのズレ
        size_dist = (rel_w - sw)**2 + (rel_h - sh)**2
        
        # 距離が小さいほどスコアが高い（単純な計算例）
        pos_score = max(0, 1.0 - (center_dist + size_dist) * 10)

        # 3. 行列数のチェック
        ratio = min(self.expected_rows, table.row_count) / max(self.expected_rows, table.row_count)
        max_col = table.col_max_count
        ratio = ratio * min(self.expected_max_cols, max_col) / max(self.expected_max_cols, max_col)

        # 位置の一致度を重視しつつ、セル数も加味する重み付け
        final_score = (pos_score * 0.7) + (ratio * 0.3)
        logger.debug(f"match(schema={self.name}, table={table.id}) result: final_score={final_score}, pos_score={pos_score}, ratio={ratio}")
        
        return final_score

    @classmethod
    def create_schema_from_table(cls, table: Table, image_shape: Tuple[int, int], name: str = "Unknown") -> 'TableSchema':
        """
        検出結果(Table)を元に、正規化されたSchemaを作成する
        """
        w_img, h_img = image_shape[:2]
        x, y, w, h = table.rect
        
        # 相対座標へ変換
        relative_rect = (x / w_img, y / h_img, w / w_img, h / h_img)

        # スキーマを作成する    
        table_schema =cls(name=name, relative_rect=relative_rect)
        table_schema.add_cells_from_table(table)

        return table_schema




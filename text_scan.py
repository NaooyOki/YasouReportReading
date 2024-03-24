import cv2
import os
import io
import sys
import re
import numpy as np
from pathlib import Path
from google.cloud import vision
from google.oauth2 import service_account
import utility as util
import json
import pickle
from dataclasses import dataclass, field, asdict
from typing import ClassVar

@dataclass
class ScanedTextInfo:
    x: int
    y: int
    width: int
    height: int
    text: str
    

class PlantNameInfo:
    def __init__(self, name, pos_x, pos_y) -> None:
        self.name = name
        self.x = pos_x
        self.y = pos_y
    
class PlantNameListInfo:
    def __init__(self, num, span) -> None:
        self.plants = []
        self.num = num
        self.span = span
    
    def add(self, info:PlantNameInfo):
        self.plants.append(info)


def isKatakana(text) -> bool:
    ret = re.match(r'^[\u30A0-\u30FF]+$', text) is not None
    return ret

def getSymbolBound(symbol):
    vertices = symbol.bounding_box.vertices
    x = vertices[0].x
    y = vertices[0].y
    w = vertices[2].x - vertices[0].x
    h = vertices[2].y - vertices[0].y
    return(x, y, w, h)

def strSymbolBound(symbol):
    (x, y, w, h) = getSymbolBound(symbol)
    str = f"({x},{y})x({w},{h})"
    return str

def ReadPlantsFromFile(img_file:str, rows:int): 
    """
    植物名が書かれた表の画像ファイルを文字認識して、種名リストを返す
    @param img_file:str 画像ファイル名
    @param rows:int     行数
    @return list[rows]   種名リスト
    """
    # Loads the image into memory
    # 画像ファイルを読み込み、バイト列に変換
    cache_file = img_file + ".picke"
    img = cv2.imread(img_file)
    namelist = ReadPlantsFromImage(img, rows, cache_file)
    return(namelist)

def ReadPlantsFromImage(img:cv2.Mat, rows:int, cache_file:str): 
    """
    植物名が書かれた表の画像ファイルを文字認識して、種名リストを返す
    @param img_file:str 画像ファイル名
    @param rows:int     行数
    @return list[rows]   種名リスト
    """
    # テキスト読み込み用オブジェクトを用意して、画像を読み込ませる
    img_reader = VisonImgTextReader()
    
    if (os.path.exists(cache_file)):
        img_reader.load_file(cache_file)
    else:
        img_reader.read_image(img)
        img_reader.save_file(cache_file)
    
    height = img.shape[0]
    width = img.shape[1]
    cell_pitch = height / rows

    # レスポンスからテキストデータを抽出
    namelist = []
    for row in range(rows):
        name = img_reader.extract_text_from_region(0, row*cell_pitch, width, cell_pitch)
        print(f"plant[{row+1}] = {name}")
        namelist.append(name)

    return(namelist)

class VisonImgTextReader:
    def __init__(self) -> None:
        self.response = None

    def read_image(self, image):
        encoded_image = cv2.imencode('.jpg', image)[1].tobytes()

        client = vision.ImageAnnotatorClient()
        vision_image = vision.Image(content=encoded_image)

        # Performs label detection on the image file
        self.response =  client.document_text_detection(
                image=vision_image,
                image_context={'language_hints': ['ja']}
            )
        
    def read_file(self, file_path):
        cv2_image = cv2.imread(file_path)
        if cv2_image is not None:
            self.read_image(cv2_image)


    def save_file(self, file_path):
        print(f"reponse type = {type(self.response)}")
        #json_str = self.response.to_json()
        #data = json.loads(json_str)
        #data = dict(self.response)

        with open(file_path, "wb") as f:           
            pickle.dump(self.response, f)
    
    def load_file(self, file_path):
        with open(file_path, "rb") as f:
            self.response = pickle.load(f)
        print(f"reponse type = {type(self.response)}")

        #self.response = vision.types.Document()
        #self.response.ParseDict(data)

    def extract_text_from_region(self, x, y, width, height):
        """
        指定された矩形領域内のテキストを抽出します。
        Args:
            x: 左上の X 座標
            y: 左上の Y 座標
            width: 幅
            height: 高さ
        Returns:
            抽出されたテキスト
        """

        text_info_list = []
        for page in self.response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        # 座標内に含まれる単語のみ抽出
                        if self.is_in_region(word.bounding_box, x, y, width, height):
                            text = ""
                            for symbol in word.symbols:
                                if (symbol.confidence >= 0.25):
                                    text += symbol.text
                                else:
                                    text += '?'
                            text_info = ScanedTextInfo(word.bounding_box.vertices[0].x, 
                                                       word.bounding_box.vertices[0].y, 
                                                       word.bounding_box.vertices[2].x - word.bounding_box.vertices[0].x, 
                                                       word.bounding_box.vertices[2].y - word.bounding_box.vertices[0].y,
                                                       text
                                                       )
                            text_info_list.append(text_info)
        
        text_info_list = sorted(text_info_list, key=lambda info: info.x)
        text = ""
        for info in text_info_list:
            text += info.text

        return text

    def is_in_region(self, bounding_box, x, y, width, height):
        """
        指定された座標内に矩形が含まれているかを判断します。
        Args:
            bounding_box: 検出したテキスト領域を表す `vision.types.BoundingBox` オブジェクト
            x, y, width, height: 対象の矩形領域
        """

        return(
            (x <= (bounding_box.vertices[0].x + bounding_box.vertices[2].x)/2 <= x+width)
            and (y <= (bounding_box.vertices[0].y + bounding_box.vertices[2].y)/2 <= y+height)
            )

# main
if __name__ == '__main__':
    img_file = "./record/202403/202403B01.JPG"
    json_file = "./record/202403/202403B01.pickle"

    img_reader = VisonImgTextReader()
    if (os.path.exists(json_file)):
        img_reader.load_file(json_file)
    else:
        img_reader.read_file(img_file)
        img_reader.save_file(json_file)

    text = img_reader.extract_text_from_region(100, 1300, 500, 100)
    print(f"text={text}")






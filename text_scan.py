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
    img = cv2.imread(img_file)
    namelist = ReadPlantsFromImage(img, rows)
    return(namelist)

def ReadPlantsFromImage(img:cv2.Mat, rows:int): 
    """
    植物名が書かれた表の画像ファイルを文字認識して、種名リストを返す
    @param img_file:str 画像ファイル名
    @param rows:int     行数
    @return list[rows]   種名リスト
    """
    # Loads the image into memory
    # 画像イメージをバイト列に変換
    height = img.shape[0]
    width = img.shape[1]
    img_bytes = cv2.imencode('.jpg', img)[1].tobytes()

    # 身元証明書のjson読み込み
    # 各自でGoogle Visionに登録して、サービスアカウントを作成し、鍵ファイルを環境変数でセットアップすること
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=img_bytes)

    # Performs label detection on the image file
    response =  client.document_text_detection(
            image=image,
            image_context={'language_hints': ['ja']}
        )

    # レスポンスからテキストデータを抽出
 
    plants = []
    plantsinfo = [[] for i in range(rows)]
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                prev_wordbox = None
                for word in paragraph.words:
                    name = ""
                    for symbol in word.symbols:
                        if (isKatakana(symbol.text) & (symbol.confidence >= 0.25)):
                            name += symbol.text
                        else:
                            strPos = strSymbolBound(symbol)
                            print(f"warning: 読み込んだ文字({symbol.text}, {strPos}) は、信頼性({symbol.confidence})が低いか、カタカナでないため、取り込みませんでした。 ")
                            name += "?"
                    plant = PlantNameInfo(name, (word.bounding_box.vertices[0].x + word.bounding_box.vertices[2].x)/2, (word.bounding_box.vertices[0].y + word.bounding_box.vertices[2].y)/2)
                    row = int(plant.y / (height / rows))
                    # print(f"name:{name}, conf:{word.confidence}, row:{row}, x:{plant.x}")
                    plantsinfo[row].append(plant)

    namelist = []
    for (row, plant) in enumerate(plantsinfo):
        plant = sorted(plant, key=lambda p: p.x)
        name = ""
        for p in plant:
            name += p.name
        namelist.append(name)

    return(namelist)



# main
if __name__ == '__main__':
    args = sys.argv
    if 2 > len(args):
        print(f"Usage {args[0]} image_file")
    else:
        util.debugTmpImgRemove()
        for i in range(1, len(args)):
            file = args[i]
            namelist = ReadPlantsFromImage(file, 30)
            print("#result")
            f = open(file.replace(".jpg", ".txt"), 'w')
            f.write(f"{file}\n")

            for i, name in enumerate(namelist):
                # print(f"{i+1}:{name}")
                f.write(f"{i+1},{name}\n")  
            f.close()





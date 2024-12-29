
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import cv2
import numpy as np
import pandas as pd
import math
import json
import glob
import csv
import os
import sys
import inspect

from dataclasses import dataclass, field, asdict
from typing import ClassVar
from dataclasses_json import dataclass_json, config
from marshmallow import Schema, fields

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from .mark import *
from .util.utility import *
import matplotlib.pyplot as plt

train_data_dir = "teachdata"
train_img_size = 32

def find_clusters_in_image(img):
  height, width = img.shape
  char_centers = []
  start_pos = 0
  stat = 0

  for x in range(int(width * 0.1), int(width * 0.9)):
    white_count = 0
    for y in range(int(height * 0.1), int(height * 0.9)):
      if img[y, x] > 80:
        white_count += 1

    if (white_count > 0):
      if (stat == 0):
         stat = 1
         start_pos = x
    else:
      if (stat == 1):
        stat = 0
        char_center = (start_pos + x)//2
        char_centers.append(char_center)

  return char_centers


def resize_image_maxpooling(img):
  """
  64x64の画像を32x32にリサイズする関数

  Args:
    img: 64x64のグレースケール画像のNumPy配列

  Returns:
    32x32にリサイズされた画像のNumPy配列
  """

  # 画像の形状を確認
  assert img.shape == (64, 64), "Image shape must be (64, 64)"

  # 32x32の空の配列を作成
  resized_img = np.zeros((32, 32))

  # 2x2の領域ごとに最大値を求め、新しい配列に格納
  for i in range(0, 64, 2):
    for j in range(0, 64, 2):
      resized_img[i//2, j//2] = np.max(img[i:i+2, j:j+2])

  return resized_img

import os

def get_csv_filenames_without_ext(directory_path):
  """
  指定されたディレクトリ以下のCSVファイルの名前（拡張子なし）を取得する関数

  Args:
    directory_path: 検索対象のディレクトリのパス

  Returns:
    ファイル名（拡張子なし）のリスト
  """

  csv_filenames = []
  for root, dirs, files in os.walk(directory_path):
    for file in files:
      if file.endswith('.csv'):
        csv_filenames.append(os.path.splitext(file)[0])
  return csv_filenames



def calc_train_param():
    # 引数の読み取り処理
    args = sys.argv
    if 3 < len(args):
        print(f"Usage {args[0]} trainfile")
        exit(-1)
    for i in range(1, len(args)):
        arg = args[i]
        files = glob.glob(arg)

    x_data = []
    y_data = []

    files = get_csv_filenames_without_ext("teachdata")

    for file in files:
        report_info = read_statreport_file("teachdata", file)

        # 画像データはNumPy配列で、shapeは(サンプル数, 64, 380, 1)
        # 教師データはNumPy配列で、shapeは(サンプル数, 4)
        # 実際の読み込み方法はデータの形式によって変わる

        stat_sub_img_width = 64
        stat_sub_clusters = find_clusters_in_image(report_info.records[0].stat_image)
        for record in report_info.records:
            x1_pos = int(stat_sub_clusters[0] - stat_sub_img_width/2)
            stat_image1 = record.stat_image[0:64, x1_pos:x1_pos+stat_sub_img_width]
            stat_image1 = resize_image_maxpooling(stat_image1)
            stat_image1 = stat_image1.reshape(1, 32, 32, 1)
            x_data.append(stat_image1)
            y_data.append(int(record.stat_tubomi/2))

            x1_pos = int(stat_sub_clusters[1] - stat_sub_img_width/2)
            stat_image1 = record.stat_image[0:64, x1_pos:x1_pos+stat_sub_img_width]
            stat_image1 = resize_image_maxpooling(stat_image1)
            stat_image1 = stat_image1.reshape(1, 32, 32, 1)
            x_data.append(stat_image1)
            y_data.append(int(record.stat_flower/2))

            x1_pos = int(stat_sub_clusters[2] - stat_sub_img_width/2)
            stat_image1 = record.stat_image[0:64, x1_pos:x1_pos+stat_sub_img_width]
            stat_image1 = resize_image_maxpooling(stat_image1)
            stat_image1 = stat_image1.reshape(1, 32, 32, 1)
            x_data.append(stat_image1)
            y_data.append(int(record.stat_seed/2))

            x1_pos = int(stat_sub_clusters[3] - stat_sub_img_width/2)
            stat_image1 = record.stat_image[0:64, x1_pos:x1_pos+stat_sub_img_width]
            stat_image1 = resize_image_maxpooling(stat_image1)
            stat_image1 = stat_image1.reshape(1, 32, 32, 1)
            x_data.append(stat_image1)
            y_data.append(int(record.stat_houshi/2))


    # 学習用とテスト用に分割
    x = np.vstack(x_data)
    y = np.array(y_data)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    plt.imshow(x_train[0], cmap='gray')
    plt.show()

    # 教師データをone-hotエンコーディング
    #y_train = to_categorical(y_train)

    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # ここに新たな畳み込み層を追加
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)

    # モデルの保存
    model.save('my_model.h5')


# main
g_skipText = False
def make_train_main():
    # 引数の読み取り処理
    args = sys.argv
    if 3 < len(args):
        print(f"Usage {args[0]} image_file")
        exit(-1)
    for i in range(1, len(args)):
        arg = args[i]
        if (arg.startswith('--')):
            if (arg == "--skipText"):
                g_skipText = True
            else:
                print(f"Ignored invalid option: {arg}")
        else:
            files = glob.glob(arg)

    debugTmpImgRemove()
    #files = ["./record/202403/202403B01.JPG", "./record/202403/202403B02.JPG"]

    for file in files:
        print(f"読み込み処理開始:{file}")
        report = create_traindata_from_report(file)
        print(f"読み込み処理終了:{file}")
        filename, ext = os.path.splitext(os.path.basename(file))
        report.write_file("./stat_img/", filename)



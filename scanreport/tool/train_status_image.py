
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

from ..mark import *
from ..util.utility import *
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



def calc_train_param(folder_path):

    folder_path = os.path.join(folder_path, '**', '*.jpg')
    files = glob.glob(folder_path)

    x_data = []
    y_data = []

    files = get_csv_filenames_without_ext("teachdata")

    for file in files:
        report_info = read_statreport_file("teachdata", file)
        #stat_sub_img_width = 32
        #stat_sub_clusters = find_clusters_in_image(report_info.records[0].stat_image)
        for record in report_info.records:
            # 蕾、花、実、胞子の画像を切り出して、マーク結果とともに教師データに加える
            h = record.stat_image.shape[0]
            w = record.stat_image.shape[1]//4
            x_data.append(record.stat_image[0:h, 0:w].reshape(1, 32, 32, 1))
            y_data.append(record.stat_tubomi.to_int())
            x_data.append(record.stat_image[0:h, w:w*2].reshape(1, 32, 32, 1))
            y_data.append(record.stat_flower.to_int())
            x_data.append(record.stat_image[0:h, w*2:w*3].reshape(1, 32, 32, 1))
            y_data.append(record.stat_seed.to_int())
            x_data.append(record.stat_image[0:h, w*3:w*4].reshape(1, 32, 32, 1))
            y_data.append(record.stat_houshi.to_int())

            # サンプル画像とマーク結果を、教師データに加える
            #x_data.append(record.sample_image)
            #y_data.append(record.sample.to_int())

    print("Data count:", len(x_data))
    print("Ans  count:", len(y_data))

    # 学習用とテスト用に分割
    x = np.vstack(x_data)
    y = np.array(y_data)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    #plt.imshow(x_train[0], cmap='gray')
    #plt.show()

    # 教師データをone-hotエンコーディング
    #y_train = to_categorical(y_train)

    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # ここに新たな畳み込み層を追加
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)

    # モデルの保存
    model.save('my_model.keras')







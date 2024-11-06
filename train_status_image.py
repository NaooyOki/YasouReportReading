
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


from status_train_info import *
from utility import *

train_data_dir = "teachdata"
train_img_size = 32


    
if __name__ == '__main__':
    # 引数の読み取り処理
    args = sys.argv
    if 3 < len(args):
        print(f"Usage {args[0]} trainfile")
        exit(-1)
    for i in range(1, len(args)):
        arg = args[i]
        files = glob.glob(arg)

    read_statreport_file("teachdata", "Stat_24YS01")
    
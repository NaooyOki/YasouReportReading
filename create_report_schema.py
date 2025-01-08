import cv2
import numpy as np
import math
import json
import glob
import csv
import os
import sys
import scanreport.util.utility as util
import inspect
import scanreport.frame.trim_report_frame as trim

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
                    trim_img = trim.trim_paper_frame(img)
                    trim_img2 = trim.trim_inner_mark2(trim_img)
                    info = trim.getDescAreaInfo(trim_img2)
                    print(f"schema info: {info}")
                    # JSONデータをファイルに書き込み
                    with open(file.lower.replace(".jpg", "_schema.json"), "w") as f:
                        json.dump(info, f)

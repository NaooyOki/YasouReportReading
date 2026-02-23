# import sys
# from .train_status_image import calc_train_param, make_train_main
# from .trim_status_image import *

# if __name__ == '__main__':
#     args = sys.argv
#     if (args[1] == "-sample"):
#         if (len(args) != 3):
#             print(f"Usage {args[0]} -sample image_folder")
#             exit(-1)
#         make_train_main(args[2])
#     elif (args[1] == "-train"):
#         calc_train_param("./teachdata")
#     else:
#         print(f"Usage {args[0]} -sample image_folder")
#         print(f"Usage {args[0]} -train")
#         exit(-1)

import argparse
import sys
import logging
from .train_status_image import calc_train_param, make_train_main
from .trim_status_image import *

def setup_logger():
    # ログを設定
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # フォーマットを指定
    formatter = logging.Formatter('%(asctime)s, %(name)s, %(levelname)s, %(message)s')
    formatter.datefmt = '%Y-%m-%d %H:%M:%S'
    
    # コンソール出力の設定
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # ファイル出力の設定
    file_hander = logging.FileHandler('scanreport.log', encoding='utf-8')
    file_hander.setLevel(logging.DEBUG)
    file_hander.setFormatter(formatter)
    logger.addHandler(file_hander)

def main():
    # ログを設定
    setup_logger()

    # 1. パーサーの設定
    parser = argparse.ArgumentParser(description="手書き丸印の機械学習コマンド")
    
    # 2. 引数の追加
    parser.add_argument("-s", "--sample", type=str, help="指定したパスの下にあるJPGファイルから学習用のサンプルを作成する")
    parser.add_argument("-t", "--train", action="store_true", help="機械学習を実施する")

    # 3. 引数の解析
    args = parser.parse_args()

    # 4. ロジックへの受け渡し
    if (args.sample is not None):
        make_train_main(args.sample)
    elif args.train:
        calc_train_param("./teachdata")
    else:
        logger.error(parser.usage)
        exit(-1)

if __name__ == "__main__":
    main()

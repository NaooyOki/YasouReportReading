import sys
from .train_status_image import calc_train_param, make_train_main
from .trim_status_image import *

if __name__ == '__main__':
    args = sys.argv
    if (args[1] == "-sample"):
        if (len(args) != 3):
            print(f"Usage {args[0]} -sample image_folder")
            exit(-1)
        make_train_main(args[2])
    elif (args[1] == "-train"):
        calc_train_param("./teachdata")
    else:
        print(f"Usage {args[0]} -sample image_folder")
        print(f"Usage {args[0]} -train")
        exit(-1)

import sys
from .train_status_image import calc_train_param, make_train_main
from .trim_status_image import *

if __name__ == '__main__':
    args = sys.argv
    if (len(args) != 3):
        print(f"Usage {args[0]} [-sample | -train] image_folder")
        exit(-1)
    if (args[1] == "-sample"):
        make_train_main(args[2])
    elif (args[1] == "-train"):
        calc_train_param(args[2])
    else:
        print(f"Invalid option: {args[1]}")
        exit(-1)

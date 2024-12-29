if __name__ == '__main__':
    from .scan_report import main as scan_main
    from .train_status_image import calc_train_param, make_train_main
    #scan_main()
    make_train_main()
    #calc_train_param()

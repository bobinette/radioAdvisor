#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2018 RadioAdvisor
# Written by Yann Giret
# --------------------------------------------------------

import csv
import matplotlib.pyplot as plt
import os

from lib.utils.load_parameters import load_image


def viz_image(idx=5):

    db_path = os.path.join("data")
    filename = os.path.join(db_path, "menisque_%s.nii.gz" % idx)

    im = load_image(filename)

    plt.imshow(im)
    plt.show()

    return im


def parse_csv():

    with open(os.path.join("Downloads", "Menisque_Train", "menisque_train_set.csv"), "r") as f:
        raw_info = csv.reader(f, delimiter=',')

        info_dict = {}
        cmpt = 0
        for row in raw_info:
            if cmpt == 0:
                cmpt += 1
                continue
            info_dict[row[0]] = {"corne_ant": bool(int(row[1])),
                                 "corne_post": bool(int(row[2])),
                                 "orient_ant": str(row[3]),
                                 "orient_post": str(row[4]),
                                 "position": str(row[5])}

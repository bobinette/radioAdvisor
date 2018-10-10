#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2018 RadioAdvisor
# Written by Yann Giret
# --------------------------------------------------------

import csv
import os


def parse_csv():

    with open(os.path.join("menisque_train_set.csv"), "r") as f:
        raw_info = csv.reader(f, delimiter=',')

        info_dict = {}
        cmpt = 0
        for row in raw_info:
            if cmpt == 0:
                cmpt += 1
                continue
            info_dict[row[0]] = {"corne_anterieure": bool(int(row[1])),
                                 "corne_posterieure": bool(int(row[2])),
                                 "orient_anterieure": str(row[3]),
                                 "orient_posterieure": str(row[4]),
                                 "position": str(row[5])}

    return info_dict

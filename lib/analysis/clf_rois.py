#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2018 Radio Advisor
# Written by Yann Giret
# --------------------------------------------------------

import numpy as np

from lib.analysis.analyzing import get_features_ycnn


def classify_rois(im, rois, net, pxl_mean, ids):

    cls_prob = get_features_ycnn(im, rois, net, pxl_mean)
    clf_ids = ids[np.argmax(cls_prob, axis=1)]

    return clf_ids

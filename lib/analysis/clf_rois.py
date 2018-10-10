#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2018 Radio Advisor
# Written by Yann Giret
# --------------------------------------------------------

import numpy as np

from lib.analysis.analyzing import get_features_ycnn
from lib.utils.config import cfg
from tools.plot import plot_rectangle


def classify_rois(im, rois, net, pxl_mean, ids):

    # Forward rois through net to get probas
    cls_prob = get_features_ycnn(im, rois, net, pxl_mean)

    # Get classification for each menisque
    clf_ids = []
    is_broken_ids = np.in1d(ids, np.array(["None"]), invert=True)
    for idx in xrange(len(rois)):
        # First check if broken
        is_broken = np.max(cls_prob[idx][is_broken_ids]) > cfg.CLS_CONF_THRESH
        # If broken assess orientation
        if is_broken:
            clf_ids.append(ids[is_broken_ids][np.argmax(cls_prob[idx][is_broken_ids])])
        else:
            clf_ids.append("None")

    return clf_ids

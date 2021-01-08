#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2018 Radio Advisor
# Written by Yann Giret
# --------------------------------------------------------

"""
Main function to analyze an image.
"""

import numpy as np
import tools._init_paths

from lib.analysis.clf_rois import classify_rois_cds
from lib.analysis.get_rois import get_rpn_rois
from lib.utils.config import cfg
from lib.utils.load_image import load_image
from lib.utils.load_parameters import load_net
from tools.plot import plot_rectangle


np.random.seed(cfg.RNG_SEED)
PLOT = True


def analyze_image(img_path):

    """
    Analyze an image coming from the self-checkout machine
        - Extract possible region of interest (roi) using both ycnn algo
        and circle detection.
        - Classfify each roi using either matching, svm or size classification.

    Input:
        - im_path:  string      path to image 'media/analysis/im_name.jpg'
    """

    # Get parameters
    net_rpn, pxl_rpn, ids_rpn = load_net(cfg.NET_DIR_RPN, cfg.NET_NAME_RPN)
    net_clf, pxl_clf, ids_clf = load_net(cfg.NET_DIR_CLF, cfg.NET_NAME_CLF)

    # Load image
    img = np.squeeze(load_image(img_path, tile_image=False, transpose=False))
    img = np.tile(img[:, :, np.newaxis], (1, 1, 3))

    # Detect menisques
    rois, ids, scores = get_rpn_rois(img, net_rpn, pxl_rpn, ids_rpn,
                                     cfg.NMS_THRESH, cfg.NMS_THRESH_CLS, cfg.CONF_THRESH)

    # Classify each menisque
    clf_ids = classify_rois_cds(img, rois, net_clf, pxl_clf, ids_clf)

    # Plot results if needed
    if PLOT and img is not None:
        plot_rectangle(img, rois, clf_ids)

    return rois, clf_ids



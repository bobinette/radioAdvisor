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

from lib.analysis.clf_rois import classify_rois
from lib.analysis.get_rois import get_rpn_rois
from lib.utils.load_image import load_image
from lib.utils.config import cfg
from tools.cache import CacheManager
from tools.plot import plot_rectangle


np.random.seed(cfg.RNG_SEED)
PLOT = True


def analyze_image(im_path):

    """
    Analyze an image coming from the self-checkout machine
        - Extract possible region of interest (roi) using both ycnn algo
        and circle detection.
        - Classfify each roi using either matching, svm or size classification.

    Input:
        - im_path:  string      path to image 'media/analysis/im_name.jpg'
    """

    # Get parameters
    CACHE_MANAGER = CacheManager()
    net_rpn, pxl_rpn, ids_rpn = CACHE_MANAGER.get_net_rpn()
    net_clf, pxl_clf, ids_clf = CACHE_MANAGER.get_net_clf()

    # Load image
    im = load_image(im_path)

    # Detect menisques
    rois, ids, scores = get_rpn_rois(im, net_rpn, pxl_rpn, ids_rpn,
                                     cfg.NMS_THRESH, cfg.NMS_THRESH_CLS, cfg.CONF_THRESH)

    # Classify each menisque
    import ipdb; ipdb.set_trace()
    clf_ids = classify_rois(im, rois, net_clf, pxl_clf, ids_clf)

    # Plot results if needed
    if PLOT and im is not None:
        plot_rectangle(im, rois, clf_ids)

    return rois, clf_ids



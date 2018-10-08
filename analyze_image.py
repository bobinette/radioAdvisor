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

from lib.analysis.get_rois import get_rpn_rois
from lib.utils.load_parameters import load_image
from lib.utils.config import cfg
from tools.cache import CacheManager
from tools.plot import plot_rectangle


np.random.seed(cfg.RNG_SEED)
PLOT = False


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
    net, pxl_mean, ids = CACHE_MANAGER.get_net()

    # Analyze image
    im = load_image(im_path)
    rois, ids, scores = get_rpn_rois(im, net, pxl_mean, ids,
                                     cfg.NMS_THRESH, cfg.NMS_THRESH_CLS, cfg.CONF_THRESH)

    # Plot results if needed
    if PLOT and im is not None:
        plot_rectangle(im, rois)

    return rois, ids



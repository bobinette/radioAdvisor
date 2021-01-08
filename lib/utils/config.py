#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2018 Radio Advisor
# Written by Yann Giret
# --------------------------------------------------------

"""
Radio Advisor Pipeline config file.
"""

import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C


# ################ CNN ANALYSIS ################

# Size of the image we forward in the net
__C.IM_SIZE_SEG = 500  # 512  # 500

# Models and where to find them
__C.NET_DIR_RPN = "vgg_rpn_cds_20200928" # 'vgg_rpn_radio'
__C.NET_NAME_RPN = 'vgg16_faster_rcnn.caffemodel'
__C.NET_DIR_CLF = "vgg_clf_cds_20200928"
__C.NET_NAME_CLF = "vgg16.caffemodel"

__C.NET_DIR_F_CLF = 'vgg_f_clf_radio_full'
# __C.NET_DIR_F_CLF = None
__C.NET_NAME_F_CLF = 'vgg16.caffemodel'
__C.NET_DIR_O_CLF = 'vgg_o_clf_radio_full'
# __C.NET_DIR_O_CLF = 'vgg_clf_radio'
__C.NET_NAME_O_CLF = 'vgg16.caffemodel'

__C.NET_DIR_SEG = "sarco_hard_20191012_resnet101"
__C.MODEL_TYPE_SEG = "deeplabv3_resnet101"
__C.NB_CLS_SEG = 3

# Pixel mean values (BGR order) as a (1, 1, 3) array
# These are the values originally used for training VGG16
# __C.PIXEL_MEANS = np.array([[[0.22625, 0.22625, 0.22625]]])
# __C.PIXEL_MEANS = np.array([[[49.56, 49.56, 49.56]]])
# __C.PIXEL_MEANS = np.array([[[54.79, 54.79, 54.79]]])
__C.PIXEL_MEANS = np.array([[[65.98, 65.98, 65.98]]])


# ################ SEG PARAMETERS ################

__C.MIN_PROB = 1e-5
__C.MAX_PROB = 1.0
__C.SEG_SMOOTHING_METH = "max"
__C.SEG_PAIRWISE_GAUSS_XY = 1
__C.SEG_PAIRWISE_BILAT_XY = 50
__C.SEG_PAIRWISE_BILAT_RGB = 14

__C.FILTER_MASK = True
__C.MUSCLE_MIN_VAL = -29
__C.MUSCLE_MAX_VAL = 1174

__C.BORDER_DIST_MAX = 10


# ################ CLF PARAMETERS ################

# Classifcation threshold for fissure detection
__C.CLS_CONF_THRESH = 0.15


# ################ RPN PARAMETERS ################

# NMS threshold used on inter-class rpn detection
__C.NMS_THRESH = 0.4
# NMS threshold used on intra-class rpn detection
__C.NMS_THRESH_CLS = 0.3
# Confidence threshold over which we keep a rpn detection
__C.CONF_THRESH = 0.5
# NMS threshold used on RPN proposals
__C.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.RPN_PRE_NMS_TOP_N = 6000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.RPN_POST_NMS_TOP_N = 500
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.RPN_MIN_SIZE = 16
# Either to use GPU or CPU nms in the rpn python layer
__C.USE_GPU_NMS = True


# ################ GENERAL ##################

# For reproducibility
__C.RNG_SEED = 4

# The architecture to use
__C.IS_GPU = True
__C.GPU_ID = 2

# Small number
__C.EPS = 10e-14


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    if len(cfg_list) % 2 != 0:
        raise AssertionError('cfg key needs to have a matching value in the list')
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            if subkey not in d:
                raise AssertionError('Subkey not in existing ones')
            d = d[subkey]
        subkey = key_list[-1]
        if subkey not in d:
                raise AssertionError('Subkey not in existing ones')
        try:
            value = literal_eval(v)
        except Exception as e:
            # handle the case when v is a string literal
            value = v
        if not isinstance(value, type(d[subkey])):
            raise AssertionError('type {} does not match original type {}'.format(type(value), type(d[subkey])))
        d[subkey] = value

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
__C.IM_SIZE_SEG = 600  # 500
__C.IM_SIZE_SEG_FOOD = 600

# Models and where to find them
__C.NET_DIR = 'vgg_rpn_radioadvisor'
__C.NET_NAME = 'vgg16.caffemodel'

# Pixel mean values (BGR order) as a (1, 1, 3) array
# These are the values originally used for training VGG16
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

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
__C.GPU_ID = 3

# Small number
__C.EPS = 10e-14

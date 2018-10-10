#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2018 Radio Advisor
# Written by Yann Giret
# --------------------------------------------------------

import caffe
import numpy as np
import os

from lib.utils.config import cfg


def load_net(train_dir, net_name):

    """
    Load the net from prototxt and caffemodel.
    If the net has been trained using a particular pixel mean,
    we load it as well.
    """

    # Load net
    train_path = os.path.join(os.path.dirname(__file__), '..', 'models', train_dir)

    proto = os.path.join(train_path, 'pipeline.prototxt')

    model = os.path.join(train_path, net_name)
    net = caffe.Net(proto, model, caffe.TEST)

    # Load pixel mean if needed
    pixel_means = cfg.PIXEL_MEANS
    if 'mean_image.npy' in os.listdir(train_path):
        pixel_means = np.load(os.path.join(train_path, 'mean_image.npy'))

    # Load ids
    ids = np.load(os.path.join(train_path, 'ids.npy'))

    return net, pixel_means, ids


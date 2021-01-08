#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2018 Radio Advisor
# Written by Yann Giret
# --------------------------------------------------------

import caffe
import numpy as np
import os
import torch
import torchvision

from lib.utils.config import cfg


def load_net(train_dir, net_name):

    """
    Load the net from prototxt and caffemodel.
    If the net has been trained using a particular pixel mean,
    we load it as well.
    """

    # Load net
    # train_path = os.path.join(os.path.dirname(__file__), '..', 'models', train_dir)
    train_path = os.path.join("/", "data", 'radio_models', train_dir)

    proto = os.path.join(train_path, 'pipeline.prototxt')

    model = os.path.join(train_path, net_name)
    net = caffe.Net(proto, model, caffe.TEST)

    # Load pixel mean if needed
    pixel_means = cfg.PIXEL_MEANS
    if 'mean_image.npy' in os.listdir(train_path):
        pixel_means = np.load(os.path.join(train_path, 'mean_image.npy'))

    # Load ids
    ids = np.load(os.path.join(train_path, 'food_label2id.npy'))

    return net, pixel_means, ids


def load_net_pytorch(train_dir):

    """
    Load a pytorch net.
    """

    # Set the gpu device
    device = torch.device("cuda:" + str(cfg.GPU_ID) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(cfg.GPU_ID)

    # Load empty model
    model = torchvision.models.segmentation.__dict__[cfg.MODEL_TYPE_SEG](num_classes=cfg.NB_CLS_SEG)
    model = model.cuda()

    # Load trained model
    net_dir = os.path.join("/", "data", "radio_models", train_dir)
    net_path = os.path.join(net_dir, "model_best.pth")
    model.load_state_dict(torch.load(net_path, map_location=device)["state_dict"])
    model.eval()

    return model

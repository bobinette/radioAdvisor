#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2019 radioAdvisor
# Written by Yann Giret
# --------------------------------------------------------

"""
Test for a given date
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from lib.analysis.analyzing_pytorch import get_seg_map
from lib.utils.load_parameters import load_net_pytorch
from lib.utils.load_image import load_image
from lib.utils.config import cfg
from tools.data_parser import load_seg_annotation

from sklearn.metrics import mean_squared_error


def cross_val(dataset_name):

    gauss_xys = np.arange(2, 5, 1)
    bilat_xys = np.arange(50, 100, 10)
    bilat_rgbs = np.arange(10, 20, 2)
    # gauss_xys = np.arange(5, 10, 5)
    # bilat_xys = np.arange(20, 30, 10)
    # bilat_rgbs = np.arange(10, 12, 2)
    stats = {}
    for gauss_xy in gauss_xys:
        for bilat_xy in bilat_xys:
            for bilat_rgb in bilat_rgbs:
                print("testing with %s, %s, %s" % (gauss_xy, bilat_xy, bilat_rgb))
                IoUs, _ = test(dataset_name, gauss_xy, bilat_xy, bilat_rgb)
                stats["%s_%s_%s" % (gauss_xy, bilat_xy, bilat_rgb)] = np.mean(IoUs)

    return stats


def test(dataset_name, gauss_xy, bilat_xy, bilat_rgb, test_idx=None):

    # Init paths
    root_dir = os.path.join("/", "home", "yann", "radioAdvisor")
    data_dir = os.path.join(root_dir, "data", "sarco", "raw_data")
    im_dir = os.path.join(root_dir, "data", "sarco", "images")

    # Load db
    extractions_dir = os.path.join("/", "data", "train_extracts", "radio_extractions")
    db = np.load(os.path.join(extractions_dir, "imdb_val_%s.npy" % dataset_name))
    # Load image info
    xl_db = {}
    xl_db.update(load_seg_annotation(os.path.join(root_dir, "data", "sarco", "dataset_train")))
    xl_db.update(load_seg_annotation(os.path.join(root_dir, "data", "sarco", "dataset_val")))

    # Load model
    model = load_net_pytorch(cfg.NET_DIR_SEG)

    # Test images
    IoUs, areas = [], []
    for idx, (im_roidb) in enumerate(db):
        if test_idx is not None:
            im_roidb = db[test_idx]
        # print(idx, im_roidb["name"])
        # Get info
        exam_id = im_roidb["name"].split("/")[-1].split(".")[0]
        info = xl_db[exam_id]

        # Load image if needed
        im_path = os.path.join(im_dir, "%s.npy" % exam_id)
        if not os.path.exists(im_path):
            data_path = os.path.join(data_dir, info["data_file"])
            im = load_image(data_path, tile_image=True, data_type="dcm")
            np.save(im_path, im)

        # Load ground truth
        annot_path = os.path.join(data_dir, info["annot_file"])
        label = load_image(annot_path, tile_image=False, data_type="nii")

        # Segment image
        seg_map, max_map, cls_prob = get_seg_map(im_path, model, gauss_xy, bilat_xy, bilat_rgb)

        # Compute IoU
        inter_area = np.sum(seg_map * label)
        union_area = np.sum((seg_map + label) > 0)
        IoUs.append(inter_area / union_area)

        # Store areas
        areas.append([np.sum(label), np.sum(seg_map)])

    # Print stats
    print("mean iou: %s" % np.mean(IoUs))
    print("MSE areas: %s" % mean_squared_error(np.array(areas)[:, 0], np.array(areas)[:, 1]))

    return IoUs, areas







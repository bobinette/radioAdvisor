#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2020 radioAdvisor
# Written by Yann Giret
# --------------------------------------------------------

"""
Export results.
"""

import cv2
import functools
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import tqdm

from lib.analysis.analyzing_pytorch import get_seg_map
from lib.utils.load_parameters import load_net_pytorch
from lib.utils.load_image import load_image
from lib.utils.config import cfg, cfg_from_list
from tools.metrics_seg import compute_metrics


def extract_seg_map(data_name, algo_version, model=None, viz_results=False):

    # Numpy data dir
    data_root_dir = os.path.join("/", "data", "radio-datasets", data_name)
    data_dir = os.path.join(data_root_dir, "dcm_data")
    im_dir = os.path.join(data_root_dir, "images")
    if not os.path.exists(im_dir):
        os.makedirs(im_dir)
    seg_dir = os.path.join(data_root_dir, "segmentations%s" % algo_version)
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)
    viz_dir = os.path.join(data_root_dir, "results_check%s" % algo_version)
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

    # Load model
    if model is None:
        model = load_net_pytorch(cfg.NET_DIR_SEG)

    for user in tqdm.tqdm(np.sort(os.listdir(os.path.join(data_dir)))):
        if not os.path.isdir(os.path.join(data_dir, user)):
            continue
        for exam in np.sort(os.listdir(os.path.join(data_dir, user))):
            if not os.path.isdir(os.path.join(data_dir, user, exam)):
                continue
            for radio in np.sort(os.listdir(os.path.join(data_dir, user, exam))):
                if not os.path.isdir(os.path.join(data_dir, user, exam, radio)):
                    continue
                for filename in os.listdir(os.path.join(data_dir, user, exam, radio)):
                    if filename == "VERSION" or "._" in filename:
                        continue
                    # print(user)
                    # Load image
                    # im_name = "%s_%s_%s_%s" % (user, exam, radio, filename)
                    im_name = user
                    im_path = os.path.join(im_dir, "%s.npy" % im_name)
                    if not os.path.exists(im_path):
                        data_path = os.path.join(data_dir, user, exam, radio, filename)
                        im = load_image(data_path, tile_image=True, data_type="dcm")
                        np.save(im_path, im)
                    # Analyze image
                    seg_map, max_map, _ = get_seg_map(im_path, model)
                    # Save segmentation
                    seg_map_nii = nib.Nifti1Image(seg_map.astype(np.int16), np.eye(4))
                    nib.save(seg_map_nii, os.path.join(seg_dir, "%s.nii.gz" % im_name))
                    cv2.imwrite(os.path.join(viz_dir, "%s.jpeg" % im_name), 255 * seg_map)
                    # Visualize results
                    if viz_results:
                        plt.imshow(seg_map)
                        plt.show()


def cross_validation():

    # Load the networks
    model = load_net_pytorch(cfg.NET_DIR_SEG)

    # PARAMS = {"SEG_PAIRWISE_GAUSS_XY": [1, 2, 3],
    #           "SEG_PAIRWISE_BILAT_XY": [30, 50, 70],
    #           "SEG_PAIRWISE_BILAT_RGB": [13, 14, 15]}
    PARAMS = {"MUSCLE_MIN_VAL": [-29, -25, -20],
              "MUSCLE_MAX_VAL": [1100, 1175, 1250]}

    nb_test = functools.reduce(lambda x, y: x * y, [len(params_) for params_ in PARAMS.values()])
    idx_std = {param_: 0 for param_ in PARAMS.keys()}

    cross_stats = []
    for i in range(nb_test):

        is_updt = False
        cfg_list = []
        for param in PARAMS.keys():
            cfg_list += [param, PARAMS[param][idx_std[param]]]
            if not is_updt:
                if idx_std[param] < len(PARAMS[param]) - 1:
                    idx_std[param] += 1
                    is_updt = True
                else:
                    idx_std[param] = 0

        cfg_from_list(cfg_list)
        extract_seg_map("_tmp", model=model)
        ious, dices = compute_metrics(algo_version="_tmp")
        mean_iou = np.mean(ious)
        mean_dice = np.mean(dices)
        cross_stats.append({"stats": {"iou": mean_iou, "dice": mean_dice},
                            "params": cfg_list})

    return cross_stats


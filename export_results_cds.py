#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2020 radioAdvisor
# Written by Yann Giret
# --------------------------------------------------------

"""
Export results.
"""

import numpy as np
import os
import pandas as pd
import tools._init_paths

from lib.analysis.clf_rois import classify_rois_cds
from lib.analysis.get_rois import get_rpn_rois
from lib.utils.load_parameters import load_net
from lib.utils.load_image import load_image
from lib.utils.config import cfg


def export_results():

    # Init paths
    root_dir = os.path.join("/", "home", "yann", "radioAdvisor")
    data_dir = os.path.join(root_dir, "data", "cancer-du-sein", "sifem_validation")
    im_dir = os.path.join(root_dir, "data", "cancer-du-sein", "test-images")

    # Load db
    db = os.listdir(data_dir)

    # Get parameters
    net_rpn, pxl_rpn, ids_rpn = load_net(cfg.NET_DIR_RPN, cfg.NET_NAME_RPN)
    net_clf, pxl_clf, ids_clf = load_net(cfg.NET_DIR_CLF, cfg.NET_NAME_CLF)

    # Test images
    # exam_ids, exam_scores = [], []
    results = []
    for idx, filename in enumerate(db):

        if not filename.endswith(".nii.gz") or "._" in filename:
            continue
        print(idx, filename)

        # Load image if needed
        exam_id = filename.split(".")[0]
        data_path = os.path.join(data_dir, filename)
        img_path = os.path.join(im_dir, "%s.npy" % exam_id)
        if not os.path.exists(img_path):
            img = np.squeeze(load_image(data_path, tile_image=False, transpose=False))
            img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
            np.save(img_path, img)
        else:
            img = np.load(img_path)

        # Detect nodules
        rois, _, det_scores = get_rpn_rois(img, net_rpn, pxl_rpn, ids_rpn,
                                           cfg.NMS_THRESH, cfg.NMS_THRESH_CLS, cfg.CONF_THRESH)

        # Classify detected nodules
        clf_scores = np.array([[0.5, 0.5]])
        if len(rois) > 0:
            rois = rois[np.argmax(det_scores)][np.newaxis, :]
            _, clf_scores = classify_rois_cds(img, rois, net_clf, pxl_clf, ids_clf)

        # Compute area
        results.append({"examen": exam_id, "prediction": clf_scores[0, 1]})

    # Create csv
    create_csv(results)


def create_csv(results):
    """
    Create the csv file based on the args.
    """
    root_dir = os.path.join("/", "home", "yann", "radioAdvisor")
    res_dir = os.path.join(root_dir, "data", "cancer-du-sein", "results")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(res_dir, 'Radioadvisor.csv'), sep=";", index=False)

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
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os

from lib.analysis.analyzing_pytorch import get_seg_map
from lib.utils.load_parameters import load_net_pytorch
from lib.utils.load_image import load_image
from lib.utils.config import cfg


def extract_seg_map(viz_results=False):

    # Numpy data dir
    im_dir = os.path.join("data", "sarco_these", "images")
    seg_dir = os.path.join("data", "sarco_these", "segmentations")
    viz_dir = os.path.join("data", "sarco_these", "results_check")
    data_dir = os.path.join("data", "sarco_these", "dcm_data")

    # Load model
    model = load_net_pytorch(cfg.NET_DIR_SEG)

    for user in np.sort(os.listdir(os.path.join(data_dir))):
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
                    print(user)
                    # Load image
                    # im_name = "%s_%s_%s_%s" % (user, exam, radio, filename)
                    im_name = user
                    im_path = os.path.join(im_dir, "%s.npy" % im_name)
                    if not os.path.exists(im_path):
                        data_path = os.path.join(data_dir, user, exam, radio, filename)
                        im = load_image(data_path, tile_image=True, data_type="dcm")
                        np.save(im_path, im)
                    # Analyze image
                    seg_map, max_map, _ = get_seg_map(im_path, model, 1, 50, 14)
                    # Save segmentation
                    seg_map_nii = nib.Nifti1Image(seg_map.astype(np.int16), np.eye(4))
                    nib.save(seg_map_nii, os.path.join(seg_dir, "%s.nii.gz" % im_name))
                    cv2.imwrite(os.path.join(viz_dir, "%s.jpeg" % im_name), 255 * seg_map)
                    # Visualize results
                    if viz_results:
                        plt.imshow(seg_map)
                        plt.show()

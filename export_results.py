#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2019 radioAdvisor
# Written by Yann Giret
# --------------------------------------------------------

"""
Export results.
"""

import csv
import nibabel as nib
import numpy as np
import os

from lib.analysis.analyzing_pytorch import get_seg_map
from lib.utils.load_parameters import load_net_pytorch
from lib.utils.load_image import load_image, get_pixel_size
from lib.utils.config import cfg
from tools.data_parser import load_seg_annotation


def export_results(dataset_name):

    # Init paths
    root_dir = os.path.join("/", "home", "yann", "radioAdvisor")
    data_dir = os.path.join(root_dir, "data", "sarco", "raw_data_")
    im_dir = os.path.join(root_dir, "data", "sarco", "images")
    res_dir = os.path.join(root_dir, "data", "sarco", "results", "axone")

    # Load db
    db = os.listdir(data_dir)

    # Load model
    model = load_net_pytorch(cfg.NET_DIR_SEG)

    # Test images
    exam_ids, areas = [], []
    for idx, filename in enumerate(db):

        if ".dcm" not in filename:
            continue
        if "._" in filename:
            continue

        print(idx, filename)

        # Load image if needed
        exam_id = filename.split(".")[0]
        data_path = os.path.join(data_dir, filename)
        im_path = os.path.join(im_dir, "%s.npy" % exam_id)
        if not os.path.exists(im_path):
            im = load_image(data_path, tile_image=True, data_type="dcm")
            np.save(im_path, im)

        # Segment image
        seg_map, max_map, _ = get_seg_map(im_path, model, 1, 50, 14)

        # Save results
        seg_map_nii = nib.Nifti1Image(seg_map.astype(np.int16), np.eye(4))
        nib.save(seg_map_nii, os.path.join(res_dir, "%s.nii.gz" % exam_id))

        # Compute area
        pixel_size = get_pixel_size(data_path)
        areas.append(pixel_size * np.sum(seg_map))
        exam_ids.append(exam_id)

    # Create csv
    create_csv(exam_ids, areas)


def create_csv(ids, areas):
    """
    Create the csv file based on the args.
    """
    root_dir = os.path.join("/", "home", "yann", "radioAdvisor")
    res_dir = os.path.join(root_dir, "data", "sarco", "results")

    with open(os.path.join(res_dir, 'axone.csv'), 'w', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow([u'examen;superficie'])
        for id_, area in zip(ids, areas):
            writer.writerow([u'%s;%s' % (id_, ",".join(str(area).split(".")))])

#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2018 radioAdvisor
# Written by Xavier Chassin
# --------------------------------------------------------

"""
Create csv
"""

import codecs
import numpy as np
import os
import tools._init_paths

from lib.analysis.clf_rois import classify_rois_, classify_rois
from lib.analysis.get_rois import get_rpn_rois
from lib.utils.load_image import load_image
from lib.utils.config import cfg
from tools.cache import CacheManager
from tools.data_parser import parse_csv


def test(csv_name=None):

    # Get parameters
    CACHE_MANAGER = CacheManager()
    net_rpn, pxl_rpn, ids_rpn = CACHE_MANAGER.get_net_rpn()
    net_f_clf, pxl_f_clf, ids_f_clf = CACHE_MANAGER.get_net_f_clf()
    net_o_clf, pxl_o_clf, ids_o_clf = CACHE_MANAGER.get_net_o_clf()

    # Load test db
    data_dir = os.path.join("data", "test_data")
    data = {filename.split(".")[0]: [] for filename in os.listdir(data_dir) if filename.endswith(".nii.gz") and "._" not in filename}
    if csv_name is not None:
        data = parse_csv(csv_name)

    # Loop over test
    test_names = []
    test_f_scores, test_l_scores, test_o_scores = np.zeros(0), np.zeros((0, 2)), np.zeros((0, 2))
    for idx, im_name in enumerate(data.keys()):
        # DEBUG
        print idx, im_name

        # Load image
        im_dir = "test_data" if csv_name is None else "raw_data"
        im_path = os.path.join("data", im_dir, "%s.nii.gz" % im_name)
        im = load_image(im_path)

        # Detect menisques
        rois, _, _ = get_rpn_rois(im, net_rpn, pxl_rpn, ids_rpn,
                                  cfg.NMS_THRESH, cfg.NMS_THRESH_CLS, cfg.CONF_THRESH)

        # Classify each menisque
        if cfg.NET_DIR_F_CLF is not None:
            clf_ids, f_score, l_scores, o_scores = classify_rois(im, rois,
                                                                 net_f_clf, pxl_f_clf, ids_f_clf,
                                                                 net_o_clf, pxl_o_clf, ids_o_clf)
        else:
            clf_ids, f_score, l_scores, o_scores = classify_rois_(im, rois,
                                                                  net_o_clf, pxl_o_clf, ids_o_clf)

        # Store results
        test_names.append(im_name)
        test_f_scores = np.hstack((test_f_scores, f_score))
        test_l_scores = np.vstack((test_l_scores, l_scores))
        test_o_scores = np.vstack((test_o_scores, o_scores))

    create_csv(test_names, test_f_scores, test_l_scores, test_o_scores)


def create_csv(ids, f_scores, l_scores, o_scores):
    """Create the csv file based on the args.

    Each arg should be a list of size n, n being the number
    of images. Then:
    - f_scores[i]: a probability indicating broken or not
    - location[i]: 0 for antérieure, 1 for postérieure
    - o_scores[i]: an array of size 2: [<h prob>, <v prob>]
    """
    res = [u'id,Corne anterieure,Corne posterieure,Fissure,Orientation horizontale,Orientation verticale']
    for m_id, f_score, locations, orientations in zip(ids, f_scores, l_scores, o_scores):
        # ant, post = (1, 0) if location == 0 else (0, 1)
        res.append(u'%s,%s,%s,%s,%s,%s' % (m_id, locations[0], locations[1], f_score, orientations[0], orientations[1]))

    csv = u'\n'.join(res)
    with codecs.open('radioAdvisor.csv', 'w', 'utf-8') as file:
        file.write('\ufeff')  # UTF-8 BOM header
        file.write(csv)


#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2020 radioAdvisor
# Written by Yann Giret
# --------------------------------------------------------

"""
Test for a given date
"""

import numpy as np
import os
import tools._init_paths

from lib.analysis.clf_rois import classify_rois_cds, classify_rois_cds_svm
from lib.analysis.get_rois import get_rpn_rois
from lib.utils.config import cfg
from lib.utils.load_parameters import load_net
from tools.db_creation import load_diagnosis
from tools.plot import plot_rectangle

from sklearn import metrics


def test(test_idx=None):

    # Get parameters
    net_rpn, pxl_rpn, ids_rpn = load_net(cfg.NET_DIR_RPN, cfg.NET_NAME_RPN)
    net_clf, pxl_clf, ids_clf = load_net(cfg.NET_DIR_CLF, cfg.NET_NAME_CLF)
    w_b = np.load(os.path.join("data", "cancer-du-sein", "w_b.npy"))

    # Load test db
    db_dir = os.path.join("/", "data", "train_extracts", "radio_extractions")
    test_db = np.load(os.path.join(db_dir, "imdb_val_cds_rpn_20200928.npy"))

    # Init test metrics
    stats = {'ok': 0, 'cmpt': 0}
    roc = {"scores": [], "labels": []}

    # Loop over test
    for idx, im_roidb in enumerate(test_db):
        # DEBUG
        if test_idx is not None:
            im_roidb = test_db[test_idx]
        print(idx, im_roidb["name"])

        # Load image
        img = np.load(im_roidb["name"])

        # Detect nodules
        rois, ids, scores = get_rpn_rois(img, net_rpn, pxl_rpn, ids_rpn,
                                         cfg.NMS_THRESH, cfg.NMS_THRESH_CLS, cfg.CONF_THRESH)

        # Classify detected nodules
        clf_ids = ["malin"]
        if len(rois) > 0:
            rois = rois[np.argmax(scores)][np.newaxis, :]
            clf_ids, clf_scores = classify_rois_cds(img, rois, net_clf, pxl_clf, ids_clf)
            # clf_id, clf_score = classify_rois_cds_svm(img, w_b, rois, net_clf, pxl_clf, ids_clf)
            # import ipdb; ipdb.set_trace()
        else:
            print("no detected nodule")

        # Get ground truth
        gt_clf = load_diagnosis("cancer-du-sein", im_roidb["photo_id"])

        # Update stats
        stats["cmpt"] += 1
        if gt_clf == clf_ids[0]:
        # if gt_clf == clf_id:
            stats["ok"] += 1
        else:
            print([[gt_clf], [clf_ids[0]]])
        # plot_rectangle(img, rois)
        # import ipdb; ipdb.set_trace()
        # Update roc
        roc["labels"].append(int(gt_clf == "malin"))
        roc["scores"].append(clf_scores[0, 1])
        # roc["scores"].append(clf_score[0])

    # Print results
    print_results(stats)
    # Get AUC score
    compute_test_score(roc)


def print_results(stats):

    # Classification
    print("-------- CLASSIFICATION -------")
    print('well_cl is ' + str(int(stats['ok'])) + '/' + str(int(stats['cmpt'])) + \
          ' -> ' + str(float(stats['ok']) / np.max((1.0, stats['cmpt']))))

def compute_test_score(roc):

    # Compute AUC
    fpr, tpr, _ = metrics.roc_curve(roc["labels"], roc["scores"])
    score = metrics.auc(fpr, tpr)

    print("AUC: %s" % str(score))

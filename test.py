#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2018 radioAdvisor
# Written by Yann Giret
# --------------------------------------------------------

"""
Test for a given date
"""

import numpy as np
import os
import tools._init_paths

from lib.analysis.clf_rois import classify_rois
from lib.analysis.get_rois import get_rpn_rois
from lib.utils.load_image import load_image
from lib.utils.config import cfg
from tools.cache import CacheManager

from scipy.optimize import linear_sum_assignment


def test(test_idx=None):

    # Get parameters
    CACHE_MANAGER = CacheManager()
    net_rpn, pxl_rpn, ids_rpn = CACHE_MANAGER.get_net_rpn()
    net_clf, pxl_clf, ids_clf = CACHE_MANAGER.get_net_clf()

    # Load test db
    test_db = np.load(os.path.join("database", "clf", "imdb_test_clf_radio.npy"))

    # Init test metrics
    stats = {'well_cl': 0, 'bad_cl': 0, 'cmpt_cl': 0,
             'well_det': 0, 'bad_det': 0, 'fp': 0, 'cmpt_det': 0,
             'all_well': 0, 'all_cmpt': 0}

    for idx, im_roidb in enumerate(test_db):
        # DEBUG
        if test_idx is not None:
            im_roidb = test_db[test_idx]
        print idx, im_roidb["name"]

        # Load image
        im_name = im_roidb["name"].split("/")[-1].split(".")[0]
        im_path = os.path.join("data", "raw_data", "%s.nii.gz" % im_name)
        im = load_image(im_path)

        # Detect menisques
        rois, _, _ = get_rpn_rois(im, net_rpn, pxl_rpn, ids_rpn,
                                  cfg.NMS_THRESH, cfg.NMS_THRESH_CLS, cfg.CONF_THRESH)

        # Classify each menisque
        clf_ids = classify_rois(im, rois, net_clf, pxl_clf, ids_clf)

        # Evaluate results
        evaluate_results(im, im_roidb["boxes"], rois, np.array(clf_ids), stats)

    # Print results
    print_results(stats)


def evaluate_results(im, gt_roidb, pred_boxes, pred_clfs, stats):

    gt_boxes, gt_clfs = get_gt_boxes(gt_roidb)

    gt2est, est2gt = iou_assign(im, gt_boxes, pred_boxes, 0.3)

    classif_eval, print_clfs = evaluate_classif(gt2est, est2gt, gt_clfs, pred_clfs)
    detect_eval, fp = evaluate_detection(gt_clfs, pred_clfs, gt2est, est2gt)

    stats = update_stats(stats, classif_eval, detect_eval, fp, print_clfs)

    return stats


def evaluate_classif(gt2est, est2gt, gt_clfs, pred_clfs):

    classif_eval = np.zeros(len(gt_clfs)).astype(int)
    print_clfs = np.zeros((0, 2))
    for idx, gt_clf in enumerate(gt_clfs):
        idx_in_assign = np.where(gt2est == idx)[0]
        if len(idx_in_assign) > 0:
            pred_clf = pred_clfs[est2gt[idx_in_assign[0]]]
            if gt_clf == pred_clf:
                classif_eval[idx] = 1
        else:
            pred_clf = ""

        print_clfs = np.vstack((print_clfs, np.array([gt_clf, pred_clf])))

    return classif_eval, print_clfs


def evaluate_detection(gt_clfs, pred_clfs, gt2est, est2gt):

    detect_eval = np.in1d(np.arange(len(gt_clfs)), gt2est).astype(int)
    fp = np.sum(np.in1d(np.arange(len(pred_clfs)), est2gt, invert=True))

    return detect_eval, fp


def iou_assign(im, gt_boxes, est_boxes, thresh):

    """
    Assign each predicted boxes to its corresponding ground truth box based on their
    respective ious.
    The optimazition is done in order to maximize the total iou.
    """

    n_boxes_gt = len(gt_boxes)
    n_boxes_est = len(est_boxes)

    IoUs = np.zeros((n_boxes_gt, n_boxes_est))
    for idx_gt in xrange(n_boxes_gt):
        box = np.round(gt_boxes[idx_gt]).astype(int)
        gt_mask = np.zeros((im.shape[0], im.shape[1]))
        gt_mask[int(box[1]): int(box[3]) + 1, int(box[0]): int(box[2]) + 1] = 1

        for idx_est in xrange(n_boxes_est):
            box = np.round(est_boxes[idx_est]).astype(int)
            est_mask = np.zeros((im.shape[0], im.shape[1]))
            est_mask[box[1]: box[3] + 1, box[0]: box[2] + 1] = 1

            # Get IoU
            tp = float(np.sum(gt_mask * est_mask))
            fp = float(np.sum((1 - gt_mask) * est_mask))
            fn = float(np.sum(gt_mask * (1 - est_mask)))
            IoUs[idx_gt, idx_est] = tp / (tp + fp + fn)

    gt2est, est2gt = linear_sum_assignment(-IoUs)

    # Remove boxes that don't intersect enough
    for idx, id_ in enumerate(gt2est):
        if IoUs[id_, est2gt[idx]] <= thresh:
            gt2est[idx] = -1
            est2gt[idx] = -1

    return gt2est, est2gt


def get_gt_boxes(roidb):

    gt_boxes = np.zeros((0, 4)).astype(int)
    gt_clfs = []

    for box in roidb:
        box_xyxy = np.array([box["box"][0], box["box"][1],
                             box["box"][0] + box["box"][2] + 1,
                             box["box"][1] + box["box"][3] + 1])
        gt_boxes = np.vstack((gt_boxes, box_xyxy))
        gt_clfs.append(box["id"])

    return gt_boxes, np.array(gt_clfs)


def update_stats(stats, classif_eval, detect_eval, fp, print_clfs):

    # Detection stats
    stats["well_det"] += np.sum(detect_eval)
    stats["bad_det"] += len(detect_eval) - np.sum(detect_eval)
    stats["fp"] += fp
    stats["cmpt_det"] += len(detect_eval)

    is_ok = (np.sum(detect_eval) == len(detect_eval)) and (fp == 0)
    # Classification stats
    stats["well_cl"] += np.sum(classif_eval)
    stats["bad_cl"] += len(classif_eval) - np.sum(classif_eval)
    stats["cmpt_cl"] += len(classif_eval)

    # Tray stats
    is_ok = is_ok and (np.sum(classif_eval) == len(classif_eval))
    stats["all_well"] += is_ok
    stats["all_cmpt"] += 1

    # Print if needed
    if not is_ok:
        print detect_eval, fp
        print print_clfs.transpose()

    return stats


def print_results(stats):

    # Detection
    print "---------- DETECTION ----------"
    print 'well_det is ' + str(int(stats['well_det'])) + '/' + str(int(stats['cmpt_det'])) + \
          ' -> ' + str(float(stats['well_det']) / np.max((1.0, stats['cmpt_det'])))
    print 'fp is ' + str(int(stats['fp'])) + '/' + str(int(stats['cmpt_det'])) + \
          ' -> ' + str(float(stats['fp']) / np.max((1.0, stats['cmpt_det'])))

    # Classification
    print "-------- CLASSIFICATION -------"
    print 'well_cl is ' + str(int(stats['well_cl'])) + '/' + str(int(stats['cmpt_cl'])) + \
          ' -> ' + str(float(stats['well_cl']) / np.max((1.0, stats['cmpt_cl'])))
    print 'all_well is ' + str(int(stats['all_well'])) + '/' + str(int(stats['all_cmpt'])) + \
          ' -> ' + str(float(stats['all_well']) / np.max((1.0, stats['all_cmpt'])))


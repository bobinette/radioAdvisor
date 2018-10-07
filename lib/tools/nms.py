#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2018 Radio Advisor
# Written by Yann Giret
# --------------------------------------------------------

import numpy as np
from lib.utils.config import cfg


def apply_nms(rois, high_thresh=0.9, low_thresh=0.5):

    """
    Apply NMS with two different threshold.
    """

    # Get rois areas to be the score for each roi
    rois = np.maximum(0, rois)
    areas = (rois[:, 2] - rois[:, 0]) * (rois[:, 3] - rois[:, 1])
    high_nms_rois = np.hstack((rois, areas.reshape((areas.shape[0], 1))))

    # First apply nms with high_thresh, keeping the biggest
    keep_idx_high = nms(high_nms_rois, high_thresh)

    # Then apply nms with low_thresh on remaining boxes, keeping the smallest
    keep_idx = keep_idx_high.copy()
    if low_thresh > 0:
        # Apply nms
        areas = np.max(areas[keep_idx_high]) - areas[keep_idx_high]
        low_nms_rois = np.hstack((rois[keep_idx_high], areas.reshape((areas.shape[0], 1))))
        keep_idx_low = nms(low_nms_rois, low_thresh)
        # Update keep_idx
        keep_idx = keep_idx_high[keep_idx_low]

    # Convert index to boolean vector
    keep = np.zeros(rois.shape[0]).astype(int)
    keep[keep_idx] = 1
    keep = keep.astype(bool)

    # Keep the rois
    rois = rois[keep]

    return rois, keep


def nms(dets, thresh, final=False):

    """
    Perform Non Maximum Suppression based on the boxes areas.
    """

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    im_area = cfg.IM_SIZE_SEG ** 2
    if final:
        # Keep only the boxes that aren't too big
        check_big = np.where(areas / im_area < 0.80)[0]
        check_small = np.where(areas / im_area > 0.05)[0]
        check = check_big[np.in1d(check_big, check_small)]
    else:
        # check = np.where(areas / im_area < 0.80)[0]
        check = np.where(areas / im_area < 100)[0]

    x1 = x1[check]
    x2 = x2[check]
    y1 = y1[check]
    y2 = y2[check]
    scores = scores[check]
    idx_boxes = np.arange(len(areas))[check]

    # If boxes with same scores, keep the biggest
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:

        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return idx_boxes[keep]


def nms_clusters(im, dets, thresh):

    """
    Perform Non Maximum Suppression based on the boxes areas.
    """

    dets = np.maximum(0, dets)
    areas = (dets[:, 2] - dets[:, 0] + 1) * (dets[:, 3] - dets[:, 1] + 1)
    dets = np.hstack((dets, areas.reshape((areas.shape[0], 1))))

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    # If boxes with same scores, keep the biggest
    order = scores.argsort()[::-1]

    clusters = {}
    while order.size > 0:

        i = order[0]
        clusters.setdefault(i, set()).update([i])
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        cluster_inds = np.where(ovr > thresh)[0]
        keep_inds = np.where(ovr <= thresh)[0]
        clusters[i].update(list((order[cluster_inds + 1])))
        order = order[keep_inds + 1]

    return clusters

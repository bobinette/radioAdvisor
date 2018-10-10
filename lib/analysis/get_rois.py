#!/usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2018 Radio Advisor
# Written by Yann Giret
# --------------------------------------------------------

import numpy as np

from lib.tools.cpu_nms import cpu_nms
from lib.analysis.analyzing import get_features_rpn


def get_rpn_rois(im, net, pxl_mean, ids, nms_thresh, nms_thresh_cls, conf_thresh):

    # Forward image through rpn net
    scores, feats, boxes = get_features_rpn(im, net, pxl_mean)

    # Post process boxes per cls
    rpn_rois, rpn_ids, rpn_scores = rpn_post_process(im, scores, boxes, ids,
                                                     nms_thresh, nms_thresh_cls, conf_thresh)

    return rpn_rois, rpn_ids, rpn_scores


def rpn_post_process(im, scores, boxes, ids, nms_thresh, nms_thresh_cls, conf_thresh):

    max_score_per_cls = np.max(scores, axis=0)

    # Loop over classes to get confident enough detection
    dets, det_ids = np.zeros((0, 5)), []
    for cls_ind, food_id in enumerate(ids[1:]):
        cls_ind += 1
        if max_score_per_cls[cls_ind] < conf_thresh:
            continue
        boxes_this_cls = boxes[:, 4 * cls_ind: 4 * (cls_ind + 1)]
        scores_this_cls = scores[:, cls_ind]
        dets_this_cls = np.hstack((boxes_this_cls, scores_this_cls[:, np.newaxis]))
        keep = cpu_nms(dets_this_cls.astype(np.float32), nms_thresh_cls)
        dets_this_cls = dets_this_cls[keep, :]
        dets_this_cls = dets_this_cls[dets_this_cls[:, -1] >= conf_thresh]
        dets = np.vstack((dets, dets_this_cls))
        det_ids += len(dets_this_cls) * [food_id]

    # There are only to menisques in the image
    top2_idx = np.argsort(dets[:, -1])[::-1][:2]
    rpn_rois = np.round(dets[top2_idx, :4]).astype(int)
    rpn_scores = dets[top2_idx, -1]
    rpn_ids = list(np.array(det_ids)[top2_idx])

    return rpn_rois, rpn_ids, rpn_scores


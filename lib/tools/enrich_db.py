#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2018 Radio Advisor
# Written by Yann Giret
# --------------------------------------------------------

""" Enrich the Foodvisor Database with boxes either from
Selective Search algorithm or with a custom algorithm generating
boxes around a ground truth box. """


import numpy as np

from lib.tools.nms import nms


def augment_im_rois(rois, n_box):

    # Create artificial boxes around the ground truth boxes
    gen_rois_wh = augment_rois_symmetric(rois, 800)

    # Assign the boxes with their corresponding ground truth
    enriched_rois, enrich2roi = get_pos_neg_rois(rois, gen_rois_wh, n_box)

    return enriched_rois, enrich2roi


def get_pos_neg_rois(rois, gen_rois_wh, n_box):

    enriched_rois = np.zeros((0, 4))
    enrich2roi = []
    # Get the positive rois
    for idx, roi in enumerate(rois):
        # We store the roi first
        enriched_rois = np.vstack((enriched_rois, roi))
        # Get the positive rois (IoU > threshold with a positive gt_box)
        enriched_roi = get_pos_rois(roi, gen_rois_wh, 0.7, n_box)
        enriched_rois = np.vstack((enriched_rois, enriched_roi))
        enrich2roi += (len(enriched_roi) + 1) * [idx]

    return enriched_rois, np.array(enrich2roi)


def augment_rois_symmetric(rois, n_sample):

    aug_rois = np.zeros((0, 4))
    for roi in rois:
        x_min, y_min, width, height = roi
        generated_rois = generate_rois(x_min, y_min, width + 1, height + 1,
                                       n_sample, 0.1)
        aug_rois = np.vstack((aug_rois, generated_rois))

    return aug_rois


def generate_rois(x_min, y_min, w, h, nb_rois_to_generate, draw_ratio):

    x_max = x_min + w
    y_max = y_min + h
    x_min_aug = np.arange(int(x_min - w * draw_ratio), int(x_min + w * draw_ratio) + 1)
    y_min_aug = np.arange(int(y_min - h * draw_ratio), int(y_min + h * draw_ratio) + 1)
    x_max_aug = np.arange(int(x_max - w * draw_ratio), int(x_max + w * draw_ratio) + 1)
    y_max_aug = np.arange(int(y_max - h * draw_ratio), int(y_max + h * draw_ratio) + 1)

    generated_rois = []
    for id_gen in range(nb_rois_to_generate):
        new_x_min = x_min_aug[np.random.randint(len(x_min_aug))]
        new_y_min = y_min_aug[np.random.randint(len(y_min_aug))]
        new_x_max = x_max_aug[np.random.randint(len(x_max_aug))]
        new_y_max = y_max_aug[np.random.randint(len(y_max_aug))]

        generated_rois.append([new_x_min, new_y_min, new_x_max - new_x_min + 1, new_y_max - new_y_min + 1])

    return generated_rois


def apply_nms(rois, labels, nms_thresh, box_type):

    if box_type == "wh":
        areas = (rois[:, 2] + 1) * (rois[:, 3] + 1)
    elif box_type == "xy":
        areas = (rois[:, 2] - rois[:, 0] + 1) * (rois[:, 3] - rois[:, 1] + 1)
    nms_rois = np.hstack((rois, areas.reshape((areas.shape[0], 1))))
    keep = nms(nms_rois, 1, nms_thresh)
    rois = rois[keep]
    labels = labels[keep]

    return rois, labels


def get_pos_rois(gt_box, rois, pos_thresh, n_box):

    areas = rois[:, 2] * rois[:, 3]
    gt_area = float(gt_box[2] + 1) * float(gt_box[3] + 1)

    inter_areas = get_inter_area(rois, gt_box)
    union_areas = areas + gt_area - inter_areas

    IoUs = inter_areas / union_areas
    pos_labels_idx = IoUs > pos_thresh

    pos_rois = rois[pos_labels_idx]

    n_pos_rois = len(pos_rois)
    keep_idx = np.random.choice(np.arange(n_pos_rois), np.min((n_pos_rois, n_box)), replace=False)
    pos_rois = pos_rois[keep_idx]

    return pos_rois


def get_inter_area(rois, gt_box):

    x_int_min = np.maximum(rois[:, 0], gt_box[0])
    y_int_min = np.maximum(rois[:, 1], gt_box[1])
    x_int_max = np.minimum(rois[:, 0] + rois[:, 2] + 1, gt_box[0] + gt_box[2] + 1)
    y_int_max = np.minimum(rois[:, 1] + rois[:, 3] + 1, gt_box[1] + gt_box[3] + 1)

    inter_area = np.maximum((x_int_max - x_int_min), 0) * np.maximum((y_int_max - y_int_min), 0)

    return inter_area

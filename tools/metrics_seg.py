#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2020 RadioAdvisor
# Written by Yann Giret
# --------------------------------------------------------

import numpy as np
import os

from lib.utils.load_image import load_image


def compute_metrics(data_name, algo_version=""):

    gt_dir = os.path.join("data", data_name, "gt_data")
    pred_dir = os.path.join("data", data_name, "segmentations%s" % algo_version)

    ious, dices = [], []
    for filename in os.listdir(gt_dir):
        if not filename.endswith("nii.gz"):
            continue
        if filename not in os.listdir(pred_dir):
            # print(filename)
            continue
        gt_mask = load_image(os.path.join(gt_dir, filename), tile_image=False)
        pred_mask = load_image(os.path.join(pred_dir, filename), tile_image=False)
        pred_mask = pred_mask.transpose()
        ious.append(compute_mask_iou(gt_mask, pred_mask))
        dices.append(compute_mask_dice(gt_mask, pred_mask))

    return ious, dices


def compute_iou(data_name, patient):

    gt_dir = os.path.join("data", data_name, "gt_data")
    pred_dir = os.path.join("data", data_name, "segmentations")

    gt_mask = load_image(os.path.join(gt_dir, "%s.nii.gz" % patient), tile_image=False)
    pred_mask = load_image(os.path.join(pred_dir, "%s.nii.gz" % patient), tile_image=False)
    pred_mask = pred_mask.transpose()

    iou = compute_mask_iou(gt_mask, pred_mask)

    return iou


def compute_dice(data_name, patient):

    gt_dir = os.path.join("data", data_name, "gt_data")
    pred_dir = os.path.join("data", data_name, "segmentations")

    gt_mask = load_image(os.path.join(gt_dir, "%s.nii.gz" % patient), tile_image=False)
    pred_mask = load_image(os.path.join(pred_dir, "%s.nii.gz" % patient), tile_image=False)
    pred_mask = pred_mask.transpose()

    dice = compute_mask_dice(gt_mask, pred_mask)

    return dice


def compute_mask_iou(gt_mask, pred_mask):

    inter_area = np.sum(gt_mask * pred_mask)
    union_area = np.sum((gt_mask + pred_mask) > 0)
    iou = inter_area / union_area

    return iou


def compute_mask_dice(gt_mask, pred_mask):

    inter_area = np.sum(gt_mask * pred_mask)
    gt_area = np.sum(gt_mask)
    pred_area = np.sum(pred_mask)
    dice = 2 * inter_area / (gt_area + pred_area)

    return dice

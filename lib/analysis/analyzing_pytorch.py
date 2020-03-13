#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2019 Radio Advisor
# Written by Yann Giret
# --------------------------------------------------------

"""
Main function to analyze an image.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydensecrf.densecrf as dcrf
import torch

from lib.analysis.analyzing import im_list_to_blob, resize_image
from lib.utils.config import cfg
from lib.utils.load_parameters import load_net_pytorch


def get_seg_map(im_path, model, gauss_xy, bilat_xy, bilat_rgb):

    """
    Get the image segmentation.
    """

    # Load net
    if model is None:
        model = load_net_pytorch(cfg.NET_DIR_SEG)

    # Get image tensor
    if im_path.split(".")[-1] == "npy":
        img = np.load(im_path)
    else:
        img = cv2.imread(im_path)
    resized_img, _ = resize_image(img.astype(np.float32, copy=True), cfg.IM_SIZE_SEG)
    resized_img -= cfg.PIXEL_MEANS
    img_blob = im_list_to_blob([resized_img])
    img_tensor = torch.tensor(img_blob).float().cuda(non_blocking=True)

    # Forward image through the net
    output = model(img_tensor)
    cls_prob = torch.nn.functional.softmax(output["out"], dim=1).data.cpu().numpy()
    cls_score = output["out"].data.cpu().numpy()[0, :, :, :].transpose((1, 2, 0))

    # Get the segmentation map from net outputs
    max_map = (np.argmax(cls_score, axis=2) == 1).astype(np.uint8)
    if cfg.SEG_SMOOTHING_METH == "max":
        seg_map = max_map.copy()
    elif cfg.SEG_SMOOTHING_METH == "crf":
        seg_map = apply_dcrf(resized_img, cls_prob, gauss_xy, bilat_xy, bilat_rgb)
        seg_map = (seg_map == 1).astype(np.uint8)
    else:
        print("not immplemented yet")

    # Resize segmentation map to original image size
    seg_map = cv2.resize(seg_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply filter where we are suer there is no muscles
    muscle_mask = (img[:, :, 0] < cfg.MUSCLE_MAX_VAL).astype(np.uint8) *\
                  (img[:, :, 0] > cfg.MUSCLE_MIN_VAL).astype(np.uint8)
    seg_map *= muscle_mask

    return seg_map, max_map, cls_prob


def apply_dcrf(img, cls_prob, gauss_xy, bilat_xy, bilat_rgb):

    """
    Apply a dense crf to the net output.
    """
    cls_prob = np.clip(cls_prob[0, :, :, :], cfg.MIN_PROB, cfg.MAX_PROB)
    n_labels = cls_prob.shape[0]
    unaries = -np.log(cls_prob).astype(np.float32)
    unaries = unaries.reshape((n_labels, -1))

    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)
    d.setUnaryEnergy(unaries)
    d.addPairwiseGaussian(sxy=(gauss_xy, gauss_xy),
                          compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=(bilat_xy, bilat_xy),
                           srgb=(bilat_rgb, bilat_rgb, bilat_rgb),
                           rgbim=img.astype(np.uint8), compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(5)
    seg_map = np.argmax(Q, axis=0).reshape(img.shape[:2])

    return seg_map

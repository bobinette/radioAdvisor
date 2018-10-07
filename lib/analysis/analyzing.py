#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2018 Radio Advisor
# Written by Yann Giret
# --------------------------------------------------------

import cv2
import numpy as np

from lib.tools.bbox_utils import bbox_transform_inv, clip_boxes
from lib.utils.config import cfg


def get_features_rpn(im, net, pxl_mean):

    """
    Forward an image trough the network to extract regions.
    """

    # Get im blob
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= pxl_mean
    im_resized, im_scale = resize_image(im_orig, cfg.IM_SIZE_SEG)
    im_blob = im_list_to_blob([im_resized])

    # Get info blob
    info_blob = np.array([[im_blob.shape[2], im_blob.shape[3], im_scale]])

    # Store blobs
    blobs = {'data': None, 'im_info': None, 'data_rois': None}
    blobs['data'] = im_blob
    blobs['im_info'] = info_blob

    # Forward through net
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False),
                            im_info=blobs['im_info'].astype(np.float32, copy=False)).copy()

    # Extract rois and project back to original image size
    rois = net.blobs['rois'].data.copy()
    boxes = rois[:, 1:] / im_scale

    # Apply bounding-box regression deltas
    box_deltas = blobs_out['bbox_pred']
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = clip_boxes(pred_boxes, im_orig.shape)

    # Extract probs
    probs = blobs_out['cls_prob']

    # Get features
    rois_feats = net.blobs["pool5_ycnn"].data
    feats = np.zeros((0, rois_feats.shape[1]))
    for roi_feats in rois_feats:
        roi_feats = roi_feats.transpose((1, 2, 0))
        roi_feats = np.max(np.max(roi_feats, axis=0), axis=0)
        feats = np.vstack((feats, roi_feats))

    return probs, feats, pred_boxes


def resize_image(im, target_size):

    """
    Resize image keeping the aspect ratio.
    """

    im_shape = im.shape
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale


def im_list_to_blob(ims):

    """
    Convert a list of images into a network input.
    Assumes images are already prepared (means subtracted, BGR order, ...).
    """

    if len(ims) > 0:
        max_shape = np.array([im.shape for im in ims]).max(axis=0)
        num_images = len(ims)
        blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                        dtype=np.float32)
        for i in xrange(num_images):
            im = ims[i]
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
        # Move channels (axis 3) to axis 1
        # Axis order will become: (batch elem, channel, height, width)
        channel_swap = (0, 3, 1, 2)
        blob = blob.transpose(channel_swap)
    else:
        blob = np.zeros(0)

    # import ipdb; ipdb.set_trace()
    return blob


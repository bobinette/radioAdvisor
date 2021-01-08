#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2020 Radio
# Written by Yann Giret
# --------------------------------------------------------

"""
Test for a given date
"""

import numpy as np
import os
import tools._init_paths
import tqdm

from lib.analysis.analyzing import im_list_to_blob, resize_image
from lib.utils.config import cfg
from lib.utils.load_parameters import load_net
from tools.plot import plot_rectangle

from sklearn import svm


def train_svms(layer_name, pct_train=0.9, C=0.1, B=1.0, pos_weight=2.0):

    # Get parameters
    net_clf, pxl_clf, ids_clf = load_net(cfg.NET_DIR_CLF, cfg.NET_NAME_CLF)

    # Load train db
    db_dir = os.path.join("/", "data", "train_extracts", "radio_extractions")
    train_db = np.load(os.path.join(db_dir, "imdb_train_cds_clf_20200928.npy"))

    # Get train_features
    pos_feats, neg_feats = [], []
    for im_roidb in tqdm.tqdm(train_db):
        # Load image
        img = np.load(im_roidb["name"])
        # Get rois
        roi_xywh = np.array(im_roidb["boxes"][0]["box"])
        roi_xyxy = np.array([roi_xywh[0], roi_xywh[1],
                             roi_xywh[0] + roi_xywh[2],
                             roi_xywh[1] + roi_xywh[3]])
        rois_xyxy = roi_xyxy[np.newaxis, :]
        # Get label
        label = im_roidb["boxes"][0]["id"]
        # Get features
        feats = get_features_ycnn(img, rois_xyxy, net_clf, pxl_clf, layer_name)
        # Store
        if label == "benin":
            neg_feats.append(feats[0])
        else:
            pos_feats.append(feats[0])

    # Init svm
    to_train_svm = svm.LinearSVC(C=C, class_weight={1: pos_weight, -1: 1},
                                 intercept_scaling=B, verbose=1,
                                 penalty='l2', loss='l1',
                                 random_state=cfg.RNG_SEED, dual=True)

    # Train SVM
    X = np.vstack((np.array(pos_feats), np.array(neg_feats)))
    y = np.hstack((np.ones(len(pos_feats)), -np.ones(len(neg_feats))))

    # Split features between training and validation
    n_rois = len(y)
    idx_train = np.random.choice(np.arange(n_rois), int(pct_train * n_rois), replace=False)
    idx_val = np.arange(n_rois)[np.in1d(np.arange(n_rois), idx_train, invert=True)]
    X_train, X_val = X[idx_train], X[idx_val]
    y_train, y_val = y[idx_train], y[idx_val]

    import ipdb; ipdb.set_trace()
    to_train_svm.fit(X_train, y_train)
    w = to_train_svm.coef_
    b = to_train_svm.intercept_[0]

    # Evaluate newly trained svm
    scores = to_train_svm.decision_function(X_val)
    pos_scores, neg_scores = compute_errors(scores, y_val)

    return w, b


def compute_errors(scores, labels):

    pos_scores = scores[(labels > 0)]
    neg_scores = scores[(labels < 0)]

    # Compute accuracy, precision and recall
    true_pos = np.sum(pos_scores > 0)
    false_pos = np.sum(neg_scores > 0)
    true_neg = np.sum(neg_scores < 0)
    false_neg = np.sum(pos_scores < 0)

    accuracy = float(true_pos + true_neg) / float(len(labels))
    precision = float(true_pos) / float(np.max(((true_pos + false_pos), 1)))
    recall = float(true_pos) / float(np.max(((true_pos + false_neg), 1)))

    print("accuracy: %s" % accuracy)
    print("precision: %s" % precision)
    print("recall %s" % recall)

    return pos_scores, neg_scores


def get_features_ycnn(im, rois, net, pxl_mean, layer_name):

    """
    Forward an image trough the network to extract regions.
    """

    # Get im blob
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= pxl_mean
    im_resized, im_scale = resize_image(im_orig, cfg.IM_SIZE_SEG)
    im_blob = im_list_to_blob([im_resized])

    # Get rois blob
    rois = np.hstack((np.zeros((len(rois), 1)), rois))
    rois_blobs = rois.astype(np.float32, copy=False)

    blobs = {'data': None, 'rois': None}
    blobs['data'] = im_blob
    blobs['rois'] = rois_blobs

    # Forward through net
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['rois'].reshape(*(blobs['rois'].shape))
    _ = net.forward(data=blobs['data'].astype(np.float32, copy=False),
                    rois=blobs['rois'].astype(np.float32, copy=False)).copy()

    # Extract data
    feats = net.blobs[layer_name].data.copy()

    return feats

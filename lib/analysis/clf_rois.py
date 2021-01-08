#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2018 Radio Advisor
# Written by Yann Giret
# --------------------------------------------------------

import numpy as np

from lib.analysis.analyzing import get_features_ycnn
from lib.tools.enrich_db import augment_im_rois
from lib.utils.config import cfg
from tools.plot import plot_rectangle


def classify_rois_cds(im, rois, net, pxl_mean, ids):

    # Forward rois through net to get probas
    cls_prob, _ = get_features_ycnn(im, rois, net, pxl_mean)
    clf_ids = ids[np.argmax(cls_prob, axis=1)]

    return np.array(clf_ids), cls_prob


def classify_rois_cds_svm(im, w_b, rois, net, pxl_mean, ids):

    _, feats = get_features_ycnn(im, rois, net, pxl_mean)
    clf_score = (np.dot(feats, w_b[0].T) + w_b[1]).ravel()
    clf_id = "benin" if clf_score < 0 else "malin"

    return clf_id, clf_score


def classify_rois_of(im, rois, f_net, f_pxl_mean, f_ids, o_net, o_pxl_mean, o_ids, enrich=False):

    if enrich:
        # Augment rois
        enriched_rois, enrich2roi = augment_im_rois(rois, 16)

        # Forward rois through net to get probas
        f_cls_prob_enriched = get_features_ycnn(im, enriched_rois, f_net, f_pxl_mean)
        o_cls_prob_enriched = get_features_ycnn(im, enriched_rois, o_net, o_pxl_mean)

        # Get cls_prob
        # import ipdb; ipdb.set_trace()
        f_cls_prob, o_cls_prob = get_probs_from_enrich(f_cls_prob_enriched, o_cls_prob_enriched, enrich2roi)

    else:
        # Forward rois through net to get probas
        f_cls_prob = get_features_ycnn(im, rois, f_net, f_pxl_mean)
        o_cls_prob = get_features_ycnn(im, rois, o_net, o_pxl_mean)

    # Get rois localisation
    locs = get_rois_loc(rois)

    # Get classification for each menisque
    clf_ids, f_scores, l_scores, o_scores = [], [], [], []
    for idx in xrange(len(rois)):
        # Fissure
        is_broken = f_cls_prob[idx, 1] > cfg.CLS_CONF_THRESH
        f_scores.append(f_cls_prob[idx, 1])
        # Localisation
        if is_broken:
            l_scores.append([0, 1] if locs[idx] else [1, 0])
        # Orientation
        clf_id = "None"
        if is_broken:
            clf_id = o_ids[np.argmax(o_cls_prob[idx])]
            o_scores.append(list(o_cls_prob[idx]))
        clf_ids.append(clf_id)

    nb_broken = len(o_scores)
    if nb_broken == 0:
        f_score = np.min(f_scores)
        l_scores, o_scores = np.array([[0.5, 0.5]]), np.array([[0.5, 0.5]])
    else:
        f_score = np.max(f_scores)
        l_scores = np.array(l_scores)
        o_scores = np.array(o_scores)
        if nb_broken == 2:
            l_scores = l_scores[np.argmax(f_scores)][np.newaxis, :]
            o_scores = o_scores[np.argmax(f_scores)][np.newaxis, :]

    return np.array(clf_ids), f_score, l_scores, o_scores


def classify_rois(im, rois, net, pxl_mean, ids):

    # Forward rois through net to get probas
    cls_prob = get_features_ycnn(im, rois, net, pxl_mean)

    # Get rois localisation
    locs = get_rois_loc(rois)

    # Get classification for each menisque
    clf_ids, f_scores, l_scores, o_scores = [], [], [], []
    is_broken_ids = np.in1d(ids, np.array(["None"]), invert=True)
    for idx in xrange(len(rois)):
        # Fissure
        is_broken = np.max(cls_prob[idx][is_broken_ids]) > cfg.CLS_CONF_THRESH
        f_score_roi = np.max(cls_prob[idx][is_broken_ids]) if is_broken else np.min(cls_prob[idx][is_broken_ids])
        f_scores.append(f_score_roi)
        # Localisation
        if is_broken:
            l_scores.append([0, 1] if locs[idx] else [1, 0])
        # Orientation
        clf_id = "None"
        if is_broken:
            clf_id = ids[is_broken_ids][np.argmax(cls_prob[idx][is_broken_ids])]
            o_scores.append(list(cls_prob[idx][is_broken_ids]))
        clf_ids.append(clf_id)

    nb_broken = len(o_scores)
    if nb_broken == 0:
        f_score = np.min(f_scores)
        l_scores, o_scores = np.array([[0.5, 0.5]]), np.array([[0.5, 0.5]])
    else:
        f_score = np.max(f_scores)
        l_scores = np.array(l_scores)
        o_scores = np.array(o_scores)
        if nb_broken == 2:
            l_scores = l_scores[np.argmax(f_scores)]
            o_scores = o_scores[np.argmax(f_scores)]

    return np.array(clf_ids), f_score, l_scores, o_scores


def get_rois_loc(rois):

    # Get the "corne classification"
    # [0: anterieure (en haut dans l'image), 1: posterieure (en bas dans l'image)]
    locs = np.zeros(len(rois)).astype(int)
    locs[np.argmax(rois[:, 3])] = 1

    return locs


def get_probs_from_enrich(f_cls_prob_enriched, o_cls_prob_enriched, enrich2roi):

    f_cls_prob = np.zeros((0, 2))
    o_cls_prob = np.zeros((0, 2))
    for idx in np.unique(enrich2roi):
        f_cls_prob_this_roi = f_cls_prob_enriched[enrich2roi == idx]
        nb_broken = np.sum(f_cls_prob_this_roi[:, 1] > cfg.CLS_CONF_THRESH)
        if nb_broken >= 8:
            f_cls_prob = np.vstack((f_cls_prob, np.max(f_cls_prob_this_roi, axis=0)))
        else:
            f_cls_prob = np.vstack((f_cls_prob, np.mean(f_cls_prob_this_roi, axis=0)))

        o_cls_prob_this_roi = o_cls_prob_enriched[enrich2roi == idx]
        o_cls_prob = np.vstack((o_cls_prob, np.mean(o_cls_prob_this_roi, axis=0)))

    return f_cls_prob, o_cls_prob

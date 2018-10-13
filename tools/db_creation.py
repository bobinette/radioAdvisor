#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2018 Radio
# Written by Yann Giret
# --------------------------------------------------------

import json
import numpy as np
import os

from lib.utils.load_image import load_image
from tools.data_parser import get_csv_data


def create_training_db(pct_train, db_type="rpn", use_previous=True):

    # Load csv
    data = get_csv_data()

    db, ids = [], set()
    for filename in os.listdir("annotations"):
        # Skip other files
        if not filename.endswith("json"):
            continue
        with open(os.path.join("annotations", filename), "r") as f:
            im_roidb = json.load(f)

        # Change the name of file for gpu training
        im_dir = os.path.join("/home", "yann", "radioAdvisor", "data", "images")
        im_name = im_roidb["name"].split("/")[-1].split(".")[0]
        im_roidb["name"] = os.path.join(im_dir, "%s.npy" % im_name)

        # Change id to have only one class for rpn training
        clean_roidb = []
        for box in im_roidb["boxes"]:
            if box["id"] == "fail":
                continue
            if db_type == "rpn":
                box["id"] = "menisque"
            elif db_type in ["clf", "f_clf", "o_clf"]:
                box["id"] = get_clf_id(box["id"], data[im_name], db_type)
            if db_type == "o_clf" and box["id"] == "None":
                continue
            clean_roidb.append(box)
            ids.update([box["id"]])

        im_roidb["boxes"] = clean_roidb

        # Store in db
        if len(im_roidb["boxes"]):
            db.append(im_roidb)

    # Split and save db
    split_and_save_db(db, ids, pct_train, db_type, use_previous)


def get_clf_id(box_id, im_info, db_type):

    is_broken = im_info[box_id]
    if db_type in ["clf", "f_clf"]:
        clf = "None"
        if is_broken:
            pos = box_id.split("_")[-1]
            clf = im_info["orient_%s" % pos]
    elif db_type == "f_clf":
        clf = "Fissure" if is_broken else "None"

    return clf


def store_images():

    data_dir = os.path.join("data", "raw_data")
    im_dir = os.path.join("data", "images")
    for filename in os.listdir(data_dir):
        # Skip other files
        if not filename.endswith('.nii.gz'):
            continue
        # Check if file already exists
        im_name = filename.split(".")[0]
        if os.path.exists(os.path.join(im_dir, "%s.npy" % im_name)):
            continue
        # Load image
        im = load_image(os.path.join(data_dir, filename))
        # Store image
        np.save(os.path.join(im_dir, "%s.npy" % im_name), im)


def split_and_save_db(database, ids, pct_train, db_type, use_previous):

    # Get the split
    idx_train, idx_test = get_split_idx(database, pct_train, use_previous)

    # Split db
    imdb_train = list(np.asarray(database)[idx_train])
    imdb_val = []
    imdb_test = list(np.asarray(database)[idx_test])

    # Get ids
    ids = ["background"] + list(np.sort(list(ids)))

    # Get the extraction dir
    extraction_dir = os.path.join("database", db_type)
    if not os.path.exists(extraction_dir):
        os.makedirs(extraction_dir)

    # Save the database
    np.save(os.path.join(extraction_dir, "imdb_train_%s_radio.npy" % db_type), imdb_train)
    np.save(os.path.join(extraction_dir, "imdb_val_%s_radio.npy" % db_type), imdb_val)
    np.save(os.path.join(extraction_dir, "imdb_test_%s_radio.npy" % db_type), imdb_test)
    np.save(os.path.join(extraction_dir, "ids_%s_radio.npy" % db_type), ids)

    # Get im names to be able to keep the same split
    train_names = [im["name"].split("/")[-1].split(".")[0] for im in imdb_train]
    test_names = [im["name"].split("/")[-1].split(".")[0] for im in imdb_test]
    np.save(os.path.join("database", "train_names.npy"), train_names)
    np.save(os.path.join("database", "test_names.npy"), test_names)


def get_data_stats(data):

    stats = {}
    for im_name, im_info in data.iteritems():
        nb_broken = 0
        if im_info["corne_anterieure"]:
            if im_info["orient_anterieure"] == "Horizontale":
                stats.setdefault("AntHorz", []).append(im_name)
            if im_info["orient_anterieure"] == "Verticale":
                stats.setdefault("AntVert", []).append(im_name)
            nb_broken += 1
        if im_info["corne_posterieure"]:
            if im_info["orient_posterieure"] == "Horizontale":
                stats.setdefault("PostHorz", []).append(im_name)
            if im_info["orient_posterieure"] == "Verticale":
                stats.setdefault("PostVert", []).append(im_name)
            nb_broken += 1
        if nb_broken == 0:
            stats.setdefault("None", []).append(im_name)
        if nb_broken == 2:
            stats.setdefault("Both", []).append(im_name)

    return stats


def get_db_stats(db):

    stats = {}

    for idx, im in enumerate(db):
        # orientation
        ids = np.array([box["id"] for box in im["boxes"]])
        is_broken_ids = ids[np.in1d(ids, np.array("None"), invert=True)]
        boxes = np.array([box["box"] for box in im["boxes"]])
        # localisation
        # [0: anterieure (en haut dans l'image), 1: posterieure (en bas dans l'image)]
        locs = np.zeros(len(boxes))
        locs[np.argmax(boxes[:, 3])] = 1
        is_broken_locs = locs[np.in1d(ids, np.array("None"), invert=True)]
        if len(is_broken_ids) == 0:
            stats.setdefault("None", []).append(idx)
            continue
        if len(is_broken_ids) == 2:
            stats.setdefault("Both", []).append(idx)
            continue
        if is_broken_ids[0] == "Horizontale" and is_broken_locs[0] == 0:
            stats.setdefault("HorzAnt", []).append(idx)
        if is_broken_ids[0] == "Verticale" and is_broken_locs[0] == 0:
            stats.setdefault("VertAnt", []).append(idx)
        if is_broken_ids[0] == "Horizontale" and is_broken_locs[0] == 1:
            stats.setdefault("HorzPost", []).append(idx)
        if is_broken_ids[0] == "Verticale" and is_broken_locs[0] == 1:
            stats.setdefault("VertPost", []).append(idx)

    return stats


def get_split_idx(database, pct_train, use_previous):

    n_im = len(database)
    if use_previous and os.path.exists(os.path.join("database", "train_names.npy")):
        # Load train and test names
        train_names = np.load(os.path.join("database", "train_names.npy"))
        test_names = np.load(os.path.join("database", "test_names.npy"))
        # Get db names
        db_names = np.array([im_roidb["name"].split("/")[-1].split(".")[0] for im_roidb in database])
        # Get train and test idx in db
        idx_train = np.arange(n_im)[np.in1d(db_names, train_names)]
        idx_test = np.arange(n_im)[np.in1d(db_names, test_names)]
    else:
        stats = get_db_stats(database)
        idx_train, idx_test = [], []
        for clf_name, clf_idxs in stats.iteritems():
            if clf_name == "Both":
                idx_train += clf_idxs
                continue
            n_im_clf = len(clf_idxs)
            n_train_clf = int(np.round(n_im_clf * pct_train))
            if n_train_clf % 2 > 0:
                n_train_clf += 1
            idx_train_clf = np.random.choice(np.arange(n_im_clf), n_train_clf, replace=False)
            idx_test_clf = np.arange(n_im_clf)[np.in1d(np.arange(n_im_clf), idx_train_clf) < 1]
            idx_train += list(np.array(clf_idxs)[idx_train_clf])
            idx_test += list(np.array(clf_idxs)[idx_test_clf])

    return idx_train, idx_test



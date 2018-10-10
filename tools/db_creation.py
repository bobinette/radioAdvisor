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
from tools.data_parser import parse_csv


def create_training_db(pct_train, db_type="rpn", use_previous=True):

    # Load csv
    data = parse_csv()

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
            elif db_type == "clf":
                box["id"] = get_clf_id(box["id"], data[im_name])
            clean_roidb.append(box)
            ids.update([box["id"]])
        im_roidb["boxes"] = clean_roidb

        # Store in db
        db.append(im_roidb)

    # Split and save db
    split_and_save_db(db, ids, pct_train, db_type, use_previous)


def get_clf_id(box_id, im_info):

    is_broken = im_info[box_id]
    orient = "None"
    if is_broken:
        pos = box_id.split("_")[-1]
        orient = im_info["orient_%s" % pos]

    return orient


def store_images():

    data_dir = os.path.join("data", "raw_data")
    im_dir = os.path.join("data", "images")
    for filename in os.listdir(data_dir):
        # Skip other files
        if not filename.endswith('.nii.gz'):
            continue
        # Load image
        im = load_image(os.path.join(data_dir, filename))
        # Store image
        im_name = filename.split(".")[0]
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
        # Split the database into a training and validation database
        n_train = int(np.round(n_im * pct_train))
        if n_train % 2 > 0:
            n_train += 1
        idx_train = np.random.choice(np.arange(n_im), n_train, replace=False)
        idx_test = np.arange(n_im)[np.in1d(np.arange(n_im), idx_train) < 1]

    return idx_train, idx_test



#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2018 Radio
# Written by Yann Giret
# --------------------------------------------------------

import cv2
import json
import numpy as np
import os

from lib.utils.load_image import load_image


def create_rpn_training_db(pct_train):

    db = []
    for filename in os.listdir("annotations"):
        # Skip other files
        if not filename.endswith("json"):
            continue
        with open(os.path.join("annotations", filename), "r") as f:
            im_roidb = json.load(f)

        # Change id to have only one class for rpn training
        clean_roidb = []
        for box in im_roidb["boxes"]:
            if box["id"] == "fail":
                continue
            box["id"] = "menisque"
            clean_roidb.append(box)
        im_roidb["boxes"] = clean_roidb

        # Change the name of file for gpu training
        im_dir = os.path.join("/home", "yann", "radioAdvisor", "data", "images")
        im_name = im_roidb["name"].split("/")[-1].split(".")[0]
        im_roidb["name"] = os.path.join(im_dir, "%s.npy" % im_name)

        # Store in db
        db.append(im_roidb)

    # Split and save db
    split_and_save_db(db, pct_train)


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


def split_and_save_db(database, pct_train):

    # Split the database into a training and validation database
    n_im = len(database)
    n_train = int(np.round(n_im * pct_train))
    if n_train % 2 > 0:
        n_train += 1
    idx_train = np.random.choice(np.arange(n_im), n_train, replace=False)
    idx_test = np.arange(n_im)[np.in1d(np.arange(n_im), idx_train) < 1]

    # Convert db
    imdb_train = list(np.asarray(database)[idx_train])
    imdb_val = []
    imdb_test = list(np.asarray(database)[idx_test])

    # Get ids
    ids = ["background", "menisque"]

    # Get the extraction dir
    extraction_dir = os.path.join("database", "rpn")
    if not os.path.exists(extraction_dir):
        os.makedirs(extraction_dir)

    # Save the database
    np.save(os.path.join(extraction_dir, "imdb_train_radio.npy"), imdb_train)
    np.save(os.path.join(extraction_dir, "imdb_val_radio.npy"), imdb_val)
    np.save(os.path.join(extraction_dir, "imdb_test_radio.npy"), imdb_test)
    np.save(os.path.join(extraction_dir, "ids_radio.npy"), ids)

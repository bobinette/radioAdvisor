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


def loadImages():
    data_dir = os.path.join("data")
    im_names = os.listdir(os.path.join(data_dir))
    im_paths = [os.path.join(data_dir, name) for name in np.sort(im_names) if ".nii" in name and "._" not in name]
    db = [{"name": im_path, "boxes": []} for im_path in im_paths]
    return db


def loadImagesToAnnotate(user):

    with open(os.path.join("split.json")) as f:
        to_annotate = json.load(f)

    to_annotate = to_annotate[user]

    return to_annotate


def annotateImages(user):

    id2name = {"1": "corne_anterieure", "2": "corne_posterieure", "0": "fail"}

    to_annotate = loadImagesToAnnotate(user)

    annotations_dir = os.path.join("annotations")
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)

    done_annotations = os.listdir(annotations_dir)
    done_annotations = [a.split(".")[0] for a in done_annotations]

    for im_name in to_annotate:
        print im_name
        if im_name.split(".")[0] in done_annotations:
            continue
        im_path = os.path.join("data", im_name)
        im_roidb = annotateImage(im_path, id2name)
        with open(os.path.join(annotations_dir, "%s.json" % im_name.split(".")[0]), "w") as f:
            json.dump(im_roidb, f)


def annoteImagesFromFilenames(filenames):
    db = loadImages()
    db = [e for e in db if e['name'] in filenames]


def annotateImage(im_path, id2name):

    global image, refPt
    refPt = []
    image = load_image(im_path)
    cv2.namedWindow("image")

    cv2.setMouseCallback("image", click_and_crop)

    # Keep looping until the 'c' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            refPt = []

        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break

    # Close all open windows
    cv2.destroyAllWindows()

    # if there are two reference points, then crop the region of interest
    # from the image and display it
    im_roidb = {"name": im_path, "boxes": []}
    if len(refPt) > 1 and len(refPt) % 2 == 0:
        for i in range(len(refPt) / 2):

            # Ensure all box directions
            topleft = (min(refPt[2 * i][0], refPt[2 * i + 1][0]), min(refPt[2 * i][1], refPt[2 * i + 1][1]))
            bottomright = (max(refPt[2 * i][0], refPt[2 * i + 1][0]), max(refPt[2 * i][1], refPt[2 * i + 1][1]))
            refPt[2 * i] = topleft
            refPt[2 * i + 1] = bottomright

            # Box as [x_min, y_min, x_max, y_max]
            box = np.asarray([refPt[2 * i][0], refPt[2 * i][1], refPt[2 * i + 1][0], refPt[2 * i + 1][1]])

            # Label box
            id_ = labelBox(im_path, box)
            id_ = id2name[str(id_)]

            roi = convert_xy_to_wh(box)
            roi_info = {'box': list(roi), 'id': id_, 'is_background': False}
            im_roidb["boxes"].append(roi_info)

    return im_roidb


def labelBox(im_path, box):

    im = load_image(im_path)
    cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.imshow(im_path, im)
    cv2.waitKey(1000)

    # Populate box for this image
    print "Selected box at: ", box
    while True:
        label = input("What is the label for this box? ")
        if label < np.inf:
            print label
            break
        else:
            print "Label is less than: %s" % str(10)

    # close all open windows
    cv2.destroyAllWindows()

    return label


def click_and_crop(event, x, y, flags, param):

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))

    # Check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))

        # draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[len(refPt) - 2], refPt[len(refPt) - 1], (0, 255, 0), 2)
        cv2.imshow("image", image)


def convert_xy_to_wh(box):

    box_wh = np.array([box[0], box[1],
                       box[2] - box[0] + 1,
                       box[3] - box[1] + 1])

    return box_wh

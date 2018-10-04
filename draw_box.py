#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2018 Radio
# Written by Yann Giret
# --------------------------------------------------------


import cv2
import numpy as np
import os

from data_parser import load_image


def annotateImages(last_idx=None):

    id2name = {"1": "corne_anterieure", "2": "corne_posterieure"}

    data_dir = os.path.join("data")
    im_names = os.listdir(os.path.join(data_dir))
    im_paths = [os.path.join(data_dir, name) for name in np.sort(im_names) if ".nii" in name and "._" not in name]
    db = [{"name": im_path, "boxes": []} for im_path in im_paths]

    annotations_dir = os.path.join("annotations")
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)

    annotated_db, n_draw, done_names = [], 0, []
    if last_idx is not None:
        annotated_db = list(np.load(os.path.join(annotations_dir, "annotation_%s.npy" % str(last_idx))))
        done_names = [im_info["name"] for im_info in annotated_db]
        n_draw = last_idx

    for im_roidb in db:
        if im_roidb["name"] in done_names:
            continue
        print im_roidb["name"]
        im_roidb = annotateImage(im_roidb, id2name)
        annotated_db.append(im_roidb)
        n_draw += 1

        if n_draw % 5 == 0:
            np.save(os.path.join(annotations_dir, "annotation_%s.npy" % n_draw), annotated_db)

    np.save(os.path.join(annotations_dir, "annotations.npy"), annotated_db)


def annotateImage(im_roidb, id2name):

    global image, refPt
    refPt = []
    image = load_image(im_roidb["name"])
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
            id_ = labelBox(im_roidb["name"], box)
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

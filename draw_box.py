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
import sys

from data_parser import load_image


class RedoException(Exception):
    pass


class QuitException(Exception):
    pass


def jsonDefault(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


def loadImages():
    data_dir = os.path.join("data")
    im_names = [f for f in os.listdir(os.path.join(data_dir)) if f.endswith('.nii.gz')]
    db = [{
        "filename": name,
        "name": os.path.join(data_dir, name),
        "boxes": [],
    } for name in im_names]
    return db


id2name = {"1": "corne_anterieure", "2": "corne_posterieure"}


def annotateImages(last_idx=None):

    db = loadImages()

    annotations_dir = os.path.join("annotations")
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)

    annotated_db, n_draw, done_names = [], 0, []
    if last_idx is not None:
        annotated_db = list(np.load(os.path.join(annotations_dir, "annotation_%s.npy" % str(last_idx))))
        done_names = [im_info["name"] for im_info in annotated_db]
        n_draw = last_idx

    to_annotate = [e for e in db if e['name'] not in done_names]

    for im_roidb in to_annotate:
        print(im_roidb["name"])
        im_roidb = annotateImage(im_roidb)
        annotated_db.append(im_roidb)
        n_draw += 1

        if n_draw % 5 == 0:
            np.save(os.path.join(annotations_dir, "annotation_%s.npy" % n_draw), annotated_db)

    np.save(os.path.join(annotations_dir, "annotations.npy"), annotated_db)


def annotateImagesFromFilenames(filenames, redo=False):
    db = loadImages()
    db = [e for e in db if e['filename'] in filenames]

    annotations_dir = os.path.join("annotations")
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)

    annotated = [
        json.load(open(os.path.join(annotations_dir, im_info)))['name']
        for im_info in os.listdir(annotations_dir)
    ]

    to_annotate = db
    if not redo:
        to_annotate = [e for e in db if e['name'] not in annotated]

    for i, im_roidb in enumerate(to_annotate):
        print("%s - %d/%d" % (im_roidb["name"], i + 1, len(to_annotate)))
        redo = True
        while redo:
            redo = False
            try:
                im_roidb = annotateImage(im_roidb)
            except RedoException:
                redo = True

        if im_roidb:
            json.dump(
                im_roidb,
                open(os.path.join(annotations_dir, "annotation_%s.json" % im_roidb["filename"]), 'w'),
                default=jsonDefault,
                indent=2,
            )


def annotateImage(im_roidb):

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
            raise RedoException

        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break

        elif key == ord("q"):
            raise QuitException()

    # Close all open windows
    cv2.destroyAllWindows()

    if len(refPt) != 4:
        return None

    # We want to have 4 points: top left and bottom right of the two boxes
    boxes = []
    if len(refPt) == 4:

        for i in range(2):
            topLeft = (min(refPt[2 * i][0], refPt[2 * i + 1][0]), min(refPt[2 * i][1], refPt[2 * i + 1][1]))
            bottomRight = (max(refPt[2 * i][0], refPt[2 * i + 1][0]), max(refPt[2 * i][1], refPt[2 * i + 1][1]))

            boxes.append((topLeft, bottomRight))

        # Reorder the boxes to have the top one first
        boxes = sorted(boxes, key=lambda p: p[0][1])

        for i, (topLeft, bottomRight) in enumerate(boxes):
            # Box as [x_min, y_min, x_max, y_max]
            box = np.asarray([topLeft[0], topLeft[1], bottomRight[0], bottomRight[1]])
            roi = convert_xy_to_wh(box)
            roi_info = {'box': list(roi), 'id': i, 'is_background': False}
            im_roidb["boxes"].append(roi_info)

    return im_roidb


def labelBox(im_path, box):

    im = load_image(im_path)
    cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.imshow(im_path, im)
    cv2.waitKey(1000)

    # Populate box for this image
    print("Selected box at: ", box)
    while True:
        label = input("What is the label for this box? ")
        if label < np.inf:
            print(label)
            break
        else:
            print("Label is less than: %s" % str(10))

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


if __name__ == '__main__':
    participants = json.load(open(os.path.join('split.json')))
    if len(sys.argv) < 2:
        print('Tell me who you are with "python draw_box.py <letter>" with letter in %s' % list(participants.keys()))

    l = sys.argv[1]
    if l not in participants.keys():
        print('%s not in %s' % (l, list(participants.keys())))

    redo = len(sys.argv) >= 3 and (sys.argv[2] == 'all' or sys.argv[2] == 'redo')
    annotateImagesFromFilenames(participants[l], redo)

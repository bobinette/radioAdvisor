#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2020 Radio
# Written by Yann Giret
# --------------------------------------------------------

import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from lib.utils.load_image import load_image
from tools.coordinates import *

ID2NAME = {"0": "malin", "1": "benin"}


def annotated_db(folder_name, db_name, data_ext=".nii.gz"):

    db, done_ims = [], []
    save_path = os.path.join("data", folder_name, "annotations", "%s.npy" % db_name)
    if os.path.exists(save_path):
        db = list(np.load(save_path))
        done_ims = [_im["name"] for _im in db]

    for filename in os.listdir(os.path.join("data", folder_name, "raw-data")):
        if not filename.endswith(data_ext) or "._" in filename:
            continue
        img_path = os.path.join("data", folder_name, "raw-data", filename)
        if img_path in done_ims:
            continue
        db.append(annotate_im_with_points(img_path))
        np.save(save_path, db)


def annotate_im_with_points(im_path, segment_only=False):

    img = np.squeeze(load_image(im_path, tile_image=False, transpose=False))
    img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
    annotator = Annotator(img, segment_only=segment_only, id2name=ID2NAME)
    annotator.run_interface()
    cv2.destroyAllWindows()

    # Retrieve boxes & labels
    roidb = [{"box": list(obj.get_box_coords(coord_format="xywh")),
              "points": obj.pnts,
              "id": obj.id,
              "name": obj.label} for obj in annotator.objects]
    db = {"name": im_path, "boxes": roidb}

    return db


class Annotator():

    def __init__(self, img, segment_only=False, id2name=None, annotator_name='Annotation Tool'):

        # Image-related
        self.origin_img = Image(img)
        # Naming & interface
        self.id2name = id2name
        self.name = annotator_name

        self.live_img = self.origin_img.copy()
        # Initializer
        self.drawing = False
        self.segment_only = segment_only
        # Outputs
        self.objects, self.tmp_pnts, self.inclusion_idxs = [], [], []
        self.live_label, self.clip_label = None, None
        self.selected_idx, self.inclusion_idx = None, None

    def reset(self):

        self.live_img = self.origin_img.copy()
        self.drawing = False
        self.objects, self.tmp_pnts, self.inclusion_idxs = [], [], []
        self.live_label, self.clip_label = None, None
        self.selected_idx, self.inclusion_idx = None, None

    def load_annotations(self, pnts_list, labels):
        """ Load external box annotations """
        if not isinstance(pnts_list, list) or not isinstance(labels, list) or len(pnts_list) != len(labels):
            raise ValueError('Both arguments must be lists of the same length')

        self.objects = [Annotation(pnts_list[idx], labels[idx]) for idx in range(len(pnts_list))]

    def validate_live_obj(self):
        """ Update image in cache """
        self.objects.append(Annotation(self.tmp_pnts, self.live_id, self.live_label, inclusion_idx=self.inclusion_idx))
        self.tmp_pnts, self.live_label, self.live_id = [], None, None

    def refresh_live_img(self, color=(180, 180, 180)):
        """ Refresh image in cache """
        self.live_img = self.origin_img.copy()
        self.live_img.add_points(self.tmp_pnts, radius=3, color=(255, 255, 255))
        for idx in range(len(self.objects)):
            # Inclusion pointer
            if isinstance(self.inclusion_idx, int) and self.inclusion_idx == idx:
                current_color = (0, 0, 0)
            # Selector
            elif isinstance(self.selected_idx, int) and idx == self.selected_idx:
                current_color = (255, 0, 0)
            else:
                current_color = color

            # Draw the annotation
            self.live_img.add_box(self.objects[idx].get_box(), self.objects[idx].label,
                                  color=current_color, label_out=self.label_out, label_top=self.label_top)
            # Included element
            if self.objects[idx].inclusion_idx is not None:
                self.live_img.add_box(self.objects[idx].get_box(), ' *',
                                      color=current_color, label_out=self.label_out, label_top=(not self.label_top))
            self.live_img.add_points(self.objects[idx].pnts, radius=4, color=(255, 255, 255))

    def draw_object(self, event, x, y, flags, param):
        """ Mouse input handler """

        # while True:
        # key = cv2.waitKey(0) & 0xFF
        # Selecting point
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True

        # Point selected
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.tmp_pnts.append((x, y))
            self.live_img.add_points([(x, y)], radius=3, color=(0, 255, 0))
            self.live_img.show(self.name)

        # Update box when 4 points
        # if len(self.tmp_pnts) == 4:
        if len(self.tmp_pnts) >= 4 and event == cv2.EVENT_RBUTTONDOWN:  # Enter

            # Get label
            self.live_label, self.live_id = None, None
            if not self.segment_only:
                if self.clip_label is None:
                    self.live_label, self.live_id = self.label_annotation()
                else:
                    self.live_label = self.clip_label

            # Drawing a box around the region of interest
            self.validate_live_obj()
            self.refresh_live_img()
            self.live_img.show(self.name)

        elif len(self.tmp_pnts) == 0 and event == cv2.EVENT_RBUTTONDOWN:
            self.refresh_live_img()

    def run_interface(self, pos=(30, 30), label_out=True, label_top=True):

        self.label_out = label_out
        self.label_top = label_top
        self.refresh_live_img()

        cv2.namedWindow(self.name)
        cv2.moveWindow(self.name, pos[0], pos[1])

        cv2.setMouseCallback(self.name, self.draw_object)

        while True:
            self.live_img.show(self.name)
            key = cv2.waitKey(0) & 0xFF

            if key == ord("r"):
                self.reset()

            # Validation
            elif key in [ord('\n'), ord('\r')]:  # Enter
                self.live_img = self.origin_img.copy()
                self.drawing = False
                self.clip_label, self.selected_idx = None, None
                break

            # Box switch
            elif key in [ord('\x00'), ord('\xe1'), ord('\xe2'), ord('$')]:  # left & right Shift & $
                if self.selected_idx is None:
                    self.selected_idx = len(self.objects) - 1
                else:
                    self.selected_idx = (self.selected_idx - 1) % len(self.objects)
                # Update selector
                self.refresh_live_img()

            # Reset current extreme selection
            elif key in [ord('\b'), ord('\r'), ord('\x08'), ord('\x7f')] and len(self.tmp_pnts) < 4 and self.selected_idx is None:  # Backspace or Delete
                self.tmp_pnts = []
                self.refresh_live_img()

            # Annotation deletion
            elif key in [ord('\b'), ord('\r'), ord('\x08'), ord('\x7f')] and self.selected_idx is not None:  # Backspace or Delete
                # Update annotations
                del self.objects[self.selected_idx]
                self.selected_idx = None
                # Refresh image
                self.live_img = self.origin_img.copy()
                self.refresh_live_img()

            # Copy label
            elif key == ord('c') and self.selected_idx is not None:
                self.clip_label = self.objects[self.selected_idx].label

            # Object inclusion
            elif key == ord('i') and self.selected_idx is not None:
                self.inclusion_idx = self.selected_idx
                self.refresh_live_img()

            # Clear selector and clip label
            elif key == ord('x'):
                self.clip_label, self.selected_idx, self.inclusion_idx = None, None, None
                self.refresh_live_img()

            # Edit label
            elif key == ord('e') and self.selected_idx is not None:
                if isinstance(self.inclusion_idx, int):
                    self.objects[self.selected_idx].edit(inclusion_idx=self.inclusion_idx)
                else:
                    if self.clip_label is None:
                        current_label = get_label(self.label_choices)
                    else:
                        current_label = self.clip_label
                    self.objects[self.selected_idx].edit(label=current_label)

                self.live_img = self.origin_img.copy()
                self.refresh_live_img()

            elif key == ord('k'):  # Kill
                # Kill
                cv2.destroyAllWindows()
                sys.exit()

        cv2.destroyWindow(self.name)

    def label_annotation(self):

        # Populate box for this image
        label = input("What is the label for this box? ")
        seg_name, segment_id = label, label
        if label in self.id2name.keys():
            seg_name = self.id2name[label]
            segment_id = self.id2name[label]

        return seg_name, segment_id


class Image():

    def __init__(self, img):
        self.img = img

    def add_box(self, box, label=None, line_width=3, font_size=1, color=(0, 0, 255), label_out=True, label_top=True, box_coord_format=None):

        # Find diagonal corners
        top_left, bot_right = get_diag_corners(box, box_coord_format)
        top_left = (int(top_left[0]), int(top_left[1]))
        bot_right = (int(bot_right[0]), int(bot_right[1]))
        # Draw the box
        cv2.rectangle(self.img, top_left, bot_right, color, line_width)

        # Label options
        if label:
            # Label position
            if label_top:
                if label_out:
                    y_pos = top_left[1] - 2 * line_width
                else:
                    y_pos = top_left[1] + 8 * line_width
            else:
                if label_out:
                    y_pos = bot_right[1] + 8 * line_width
                else:
                    y_pos = bot_right[1] - 2 * line_width

            lab_pos = (top_left[0], y_pos)
            if isinstance(label, list):
                # Multi- label
                for i, line in enumerate(label):
                    y = int(y_pos + 30 * i * font_size)
                    cv2.putText(self.img, str(line).encode('utf-8'), (top_left[0], y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_size, color, int(line_width * font_size))
            else:
                # import ipdb; ipdb.set_trace()
                cv2.putText(self.img, label, lab_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, font_size, color, int(line_width * font_size))

                # cv2.putText(im_debug, food_id, (np.min(points[:, 0]), np.min(points[:, 1])),
                #     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def add_boxes(self, boxes, labels=None, line_width=3, font_size=1, color=(0, 0, 255), label_out=True, label_top=True, box_coord_format=None):

        if labels and len(boxes) != len(labels):
            raise ValueError('arguments boxes and labels don\'t have the same length')

        # Multi color
        if isinstance(color, tuple):
            colors = [color] * len(boxes)
        elif isinstance(color, list):
            colors = color
        else:
            raise ValueError('Color argument must either be a tuple or a list of tuples')

        # Label
        if labels is None:
            actual_labels = [None] * len(boxes)
        elif isinstance(labels, list):
            actual_labels = labels
        else:
            raise ValueError('Labels argument must either be None or a list/nested-list of strings')

        # Draw all annotations
        for i, box in enumerate(boxes):

            self.add_box(box, actual_labels[i], line_width=line_width, font_size=font_size, color=colors[i],
                         label_out=label_out, label_top=label_top, box_coord_format=box_coord_format)

    def add_points(self, points, radius, color=(0, 0, 255)):

        for x, y in points:
            cv2.circle(self.img, (x, y), radius, color, -1)

    def show(self, win_name, pos=None):
        cv2.imshow(win_name, self.img)
        if isinstance(pos, tuple):
            cv2.moveWindow(win_name, pos[0], pos[1])

    def show_plt(self):
        plt.imshow(self.img[:, :, (2, 1, 0)])
        plt.show()

    def copy(self):
        return copy.deepcopy(Image(self.img))

    def export(self, save_path):
        cv2.imwrite(save_path, self.img)


class Annotation():

    def __init__(self, pnts_coord, segment_id, label, inclusion_idx=None):
        self.label = label
        self.id = segment_id
        self.pnts = pnts_coord
        self.inclusion_idx = inclusion_idx

    def edit(self, pnts_coord=None, segment_id=None, label=None, inclusion_idx=None):
        if segment_id is not None:
            self.segment_id = segment_id
        if label is not None:
            self.label = label
        if pnts_coord is not None:
            self.pnts = pnts_coord
        if inclusion_idx is not None:
            self.inclusion_idx = inclusion_idx

    def get_box(self):
        return list(get_tlbr_from_points(self.pnts))

    def get_box_coords(self, coord_format="xyxy"):
        return get_box_from_points(self.pnts, coord_format=coord_format)


def get_label(choices):
    """ Request the object's label from the user """

    # Ask for user input only if there are multiple elements in the mapping
    if len(choices) > 1:
        label_idx = input('Which label for this box?\n')
        if label_idx >= len(choices):
            # Clear previous line
            sys.stdout.write("\033[F\033[K")
            print('This is not a valid label id')
            label_idx = input('Which label for this box?\n')
        while label_idx >= len(choices):
            # Clear previous two lines
            sys.stdout.write("\033[F\033[K")
            sys.stdout.write("\033[F\033[K")
            label_idx = input('Which label for this box?\n')
    else:
        label_idx = 0

    return choices[label_idx]

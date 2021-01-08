#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2018 Radio
# Written by Yann Giret
# --------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_rectangle(im, boxes, names=None, wh=False):
    # plt.figure(figsize=(10,10))
    currentAxis = plt.gca()
    plt.imshow(im[:, :, (2, 1, 0)])
    if len(boxes.shape) == 1:
        boxes = boxes.reshape((1, 4))
    for idx, box in enumerate(boxes):
        xmin = box[0]
        ymin = box[1]
        if wh:
            width = box[2]
            height = box[3]
        else:
            width = (box[2]) - (box[0] + 1)
            height = (box[3]) - (box[1] + 1)
        currentAxis = plt.gca()
        currentAxis.add_patch(Rectangle((xmin, ymin), width, height,
                              facecolor="none", edgecolor="yellow", label="test"))
        if names is not None:
            currentAxis.annotate(names[idx], (xmin, ymin), color="black",
                                 bbox=dict(boxstyle="square", fc="yellow"))
    plt.axis('off')
    plt.show()

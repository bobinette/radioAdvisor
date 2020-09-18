#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2020 Radio
# Written by Yann Giret
# --------------------------------------------------------


def get_coord_dict(box_coord_format=None):
    """
    Get a matching dictionary for box coordinate system
    Args:
        box_coord_format (list): format of box coordinates used. possible atomic values: xmin, xmax, ymin, ymax, w, h
    Returns:
        coord_dict (dict): matching dictionaries for coordinates and array indexes
    """

    coord_dict = {'xmin': 0, 'ymin': 1, 'xmax': 2, 'ymax': 3, 'w': None, 'h': None}
    if box_coord_format is not None:
        if len(box_coord_format) != 4:
            raise ValueError('Expected an array of length 4')
        else:
            for idx, coord in enumerate(box_coord_format):
                coord_dict[coord] = idx

    return coord_dict


def get_diag_corners(box, box_coord_format=None):
    """
    Compute the top left and bottom right corner coordinates of a given box
    Args:
        box (list): 2D coordinates of the box
        box_coord_format (list): format of box coordinates used. possible atomic values: xmin, xmax, ymin, ymax, w, h
    Returns:
        top_left (tuple): 2D coordinates of the top left corner of the box
        bot_right (tuple): 2D coordinates of the bottom right corner of the box
    """

    if box_coord_format is None and len(box) == 2:
        top_left, bot_right = tuple(box[0]), tuple(box[1])
    else:
        coord_dict = get_coord_dict(box_coord_format)

        top_left = (box[coord_dict['xmin']], box[coord_dict['ymin']])

        if coord_dict['w'] is not None and coord_dict['h'] is not None:
            bot_right = (box[coord_dict['xmin']] + box[coord_dict['w']], box[coord_dict['ymin']] + box[coord_dict['h']])
        else:
            bot_right = (box[coord_dict['xmax']], box[coord_dict['ymax']])

    return top_left, bot_right


def get_tlbr_from_points(extreme_coords):
    """
    Compute the bounding box using extreme points of the object
    Args:
        extreme_coords (list(tuple)): list of 2D coordinates of extreme points
    Returns:
        top_left (tuple): 2D coordinates of the top left corner of the box
        bot_right (tuple): 2D coordinates of the bottom right corner of the box
    """

    # if not isinstance(extreme_coords, list) or len(extreme_coords) != 4:
    #     raise ValueError('Expected a list of 4 tuples')

    x_vec = [x for x, y in extreme_coords]
    y_vec = [y for x, y in extreme_coords]
    top_left = (min(x_vec), min(y_vec))
    bot_right = (max(x_vec), max(y_vec))

    return top_left, bot_right


def get_box_from_points(extreme_coords, coord_format="xyxy"):
    """
    Compute the bounding box using extreme points of the object
    Args:
        extreme_coords (list(tuple)): list of 2D coordinates of extreme points
    Returns:
        top_left (tuple): 2D coordinates of the top left corner of the box
        bot_right (tuple): 2D coordinates of the bottom right corner of the box
    """

    # if not isinstance(extreme_coords, list) or len(extreme_coords) != 4:
    #     raise ValueError('Expected a list of 4 tuples')

    x_vec = [x for x, y in extreme_coords]
    y_vec = [y for x, y in extreme_coords]
    if coord_format == "xyxy":
        box = [min(x_vec), min(y_vec), max(x_vec), max(y_vec)]
    elif coord_format == "xywh":
        box = [min(x_vec), min(y_vec), max(x_vec) - min(x_vec) + 1, max(y_vec) - min(y_vec) + 1]
    else:
        print("box format non valid")

    return box

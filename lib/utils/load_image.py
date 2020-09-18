#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2018 Radio Advisor
# Written by Yann Giret
# --------------------------------------------------------

import nibabel as nib
import numpy as np
import pydicom


def load_image(data_path, tile_image=True, transpose=True, data_type="nii"):

    """
    Extract image from .nii.gz file
    """

    # Load image
    if data_type == "nii":
        data = nib.load(data_path)
        im = data.get_data()
    elif data_type == "dcm":
        data = pydicom.dcmread(data_path, force=True)
        im = data.pixel_array
    else:
        print("data type not supported yet")

    # Tile image if needed
    if tile_image:
        if len(im.shape) == 2:
            im = np.tile(im[:, :, np.newaxis], (1, 1, 3))
        else:
            im = np.tile(im, (1, 1, 3))
    else:
        if len(im.shape) == 3:
            im = im[:, :, 0]

    if data_type == "nii" and transpose:
        im = im.transpose()

    return im


def get_pixel_size(data_path):

    """
    Extract image from .nii.gz file
    """

    # Load image
    data = pydicom.dcmread(data_path)
    pxl_size = float(data.PixelSpacing[0]) * float(data.PixelSpacing[1])

    return pxl_size


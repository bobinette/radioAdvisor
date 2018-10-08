#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2018 Radio Advisor
# Written by Yann Giret
# --------------------------------------------------------

import nibabel as nib
import numpy as np


def load_image(data_path):

    """
    Extract image from .nii.gz file
    """

    data = nib.load(data_path)
    im = data.get_data()
    im = np.tile(im[:, :, np.newaxis], (1, 1, 3))

    return im

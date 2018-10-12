#!usr/bin/python
# -*- coding: utf-8 -*-

from collections import defaultdict
import json
import os

# --------------------------------------------------------
# Copyright (c) 2018 RadioAdvisor
# Written by Xavier Chassin
# --------------------------------------------------------


def split_images(participants, path, val=False):
    n = len(participants)

    attributions = defaultdict(list)
    for i, filename in enumerate(os.listdir(path)):
        # Skip other files
        if not filename.endswith('.nii.gz'):
            continue
        # Only validation set
        if val and "validation" not in filename:
            continue

        p = participants[i % n]
        attributions[p].append(filename)

    save_prefix = "val" if val else "train"
    save_name = "%s_split.json" % save_prefix
    json.dump(attributions, open(os.path.join('.', save_name), 'w'), indent=2)


if __name__ == '__main__':
    split_images(['C', 'X', 'Y'], os.path.join('.', 'data'))

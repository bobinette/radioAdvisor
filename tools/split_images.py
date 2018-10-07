#!usr/bin/python
# -*- coding: utf-8 -*-

from collections import defaultdict
import json
import os

# --------------------------------------------------------
# Copyright (c) 2018 RadioAdvisor
# Written by Xavier Chassin
# --------------------------------------------------------


def split_images(participants, path):
    n = len(participants)

    attributions = defaultdict(list)
    for i, filename in enumerate(os.listdir(path)):
        # Skip other files
        if not filename.endswith('.nii.gz'):
            continue

        p = participants[i % n]
        attributions[p].append(filename)

    json.dump(attributions, open(os.path.join('.', 'split.json'), 'w'), indent=2)


if __name__ == '__main__':
    split_images(['C', 'X', 'Y'], os.path.join('.', 'data'))

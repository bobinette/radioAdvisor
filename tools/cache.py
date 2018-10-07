#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2018 Radio Advisor
# Written by Yann Giret
# --------------------------------------------------------

import tools._init_paths
import caffe

from lib.utils.config import cfg
from lib.utils.load_parameters import load_net


class CacheManager(object):
    """
    Object to manage cache through the app
    """

    NET, PXL_MEAN, IDS = None, None, None

    def get_net_seg(self):
        if self.NET is None:
            try:
                self.NET, self.PXL_MEAN, self.IDS = self.compute_net(cfg.NET_DIR, cfg.NET_NAME)
            except Exception:
                print "Unable loading net."

        return self.NET, self.PXL_MEAN, self.IDS

    def compute_net(self, train_dir, net_name):
        """
        Compute net for cache manager
        """
        if cfg.IS_GPU:
            caffe.set_device(cfg.GPU_ID)
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        return load_net(train_dir, net_name)


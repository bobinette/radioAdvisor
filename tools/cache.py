#!usr/bin/python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2018 Radio Advisor
# Written by Yann Giret
# --------------------------------------------------------

import caffe

from lib.utils.config import cfg
from lib.utils.load_parameters import load_net


class CacheManager(object):
    """
    Object to manage cache through the app
    """

    NET_RPN, PXL_RPN, IDS_RPN = None, None, None
    NET_CLF, PXL_CLF, IDS_CLF = None, None, None

    def get_net_rpn(self):
        if self.NET_RPN is None:
            try:
                self.NET_RPN, self.PXL_RPN, self.IDS_RPN = self.compute_net(cfg.NET_DIR_RPN, cfg.NET_NAME_RPN)
            except Exception:
                print "Unable loading net."

        return self.NET_RPN, self.PXL_RPN, self.IDS_RPN

    def get_net_clf(self):
        if self.NET_CLF is None:
            try:
                self.NET_CLF, self.PXL_CLF, self.IDS_CLF = self.compute_net(cfg.NET_DIR_CLF, cfg.NET_NAME_CLF)
            except Exception:
                print "Unable loading net."

        return self.NET_CLF, self.PXL_CLF, self.IDS_CLF

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


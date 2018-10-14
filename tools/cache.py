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
    NET_F_CLF, PXL_F_CLF, IDS_F_CLF = None, None, None
    NET_O_CLF, PXL_O_CLF, IDS_O_CLF = None, None, None

    def get_net_rpn(self):
        if self.NET_RPN is None:
            try:
                self.NET_RPN, self.PXL_RPN, self.IDS_RPN = self.compute_net(cfg.NET_DIR_RPN, cfg.NET_NAME_RPN)
            except Exception:
                print "Unable loading net."

        return self.NET_RPN, self.PXL_RPN, self.IDS_RPN

    def get_net_f_clf(self):
        if self.NET_F_CLF is None:
            try:
                self.NET_F_CLF, self.PXL_F_CLF, self.IDS_F_CLF = self.compute_net(cfg.NET_DIR_F_CLF, cfg.NET_NAME_F_CLF)
            except Exception:
                print "Unable loading net."

        return self.NET_F_CLF, self.PXL_F_CLF, self.IDS_F_CLF

    def get_net_o_clf(self):
        if self.NET_O_CLF is None:
            try:
                self.NET_O_CLF, self.PXL_O_CLF, self.IDS_O_CLF = self.compute_net(cfg.NET_DIR_O_CLF, cfg.NET_NAME_O_CLF)
            except Exception:
                print "Unable loading net."

        return self.NET_O_CLF, self.PXL_O_CLF, self.IDS_O_CLF

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


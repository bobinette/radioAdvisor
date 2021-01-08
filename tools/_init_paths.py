""" Import path for Pipeline"""
import os
import os.path as osp
import sys

os.environ["GLOG_minloglevel"] = "3"


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


# Add lib to PYTHONPATH
lib_path = osp.join(osp.dirname(__file__), "..", "lib")
add_path(lib_path)

# # Add python caffe to PYTHONPATH
caffe_path = osp.join('/', 'lib_build', "caffe-faster-rcnn", "python")
add_path(caffe_path)



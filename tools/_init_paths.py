""" Import path for Pipeline"""
import os
import os.path as osp
import sys

os.environ["GLOG_minloglevel"] = "3"


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(osp.dirname(__file__), "lib")
add_path(lib_path)

# Add pygco to PYTHONPATH
pygco_path = "/home/foodvisor/gco_python"
add_path(pygco_path)

# Add ESS to PYTHONPATH
ess_path = "ESS-1_1"
add_path(ess_path)

# # Add python caffe to PYTHONPATH
# caffe_path = osp.join('/home', 'foodvisor', 'server', 'foodvisor', 'apps', 'pipeline', 'lib', 'caffe', 'python')
caffe_path = osp.join(osp.dirname(__file__), 'caffe-fast-rcnn', 'python')
add_path(caffe_path)

# py_path = '/home/foodvisor/Py-Pipeline/lib/caffe/python'
# add_path(py_path)

# # Add python caffe to PYTHONPATH

# py_path = '/home/foodvisor/Py-Pipeline/lib/caffe/python'
# add_path(py_path)

# # Add caffe to PYTHONPATH

# caffe_path = '/home/foodvisor/Py-Pipeline/lib/caffe/.build_release/lib'
# add_path(caffe_path)

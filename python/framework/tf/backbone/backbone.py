#!/usr/bin/python

# import 库
try:
    import sys
    import glog
except ImportError as e:
    print(e)
    print("please install glog first")

# 校验大版本号必须=3
PYTHON_VERSION = sys.version_info[0]
glog.check_eq(PYTHON_VERSION, 3, "only support python version 3, your version is %d" % PYTHON_VERSION)

class 

class Blob(object):
    def __init__(self, shape, data_format):
        self._shape = shape
        self._data_format = data_format

class Backbone(object):
    def __init__(self, input_blob_list):
        self._input_blob_list = input_blob_list

if __name__ == "__main__":
    shape = 
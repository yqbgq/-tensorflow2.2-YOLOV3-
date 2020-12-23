# ================================================================
#
#   Editor      : Pycharm
#   File name   : common
#   Author      : HuangWei
#   Created date: 2020-12-23 15:10
#   Email       : 446296992@qq.com
#   Description : 公共工具
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
import numpy as np


def read_class_names(class_file_name):
    """loads class name from a file"""
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def get_anchors(anchor_path):
    """loads the anchors from a file"""
    with open(anchor_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape([3, 3, 2])

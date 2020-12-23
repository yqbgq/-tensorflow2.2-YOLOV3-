# ================================================================
#
#   Editor      : Pycharm
#   File name   : dataloader
#   Author      : HuangWei
#   Created date: 2020-12-23 14:57
#   Email       : 446296992@qq.com
#   Description : 自定义数据加载器
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
import os
import tensorflow as tf
import numpy as np

import config
from utils import common


class Dataloader:
    def __init__(self, data_type):
        self.yolo_cfg = config.YoloConfig
        if data_type == "train":
            self.cfg = config.TrainConfig
        else:
            self.cfg = config.TestConfig

        self.annotation_path = self.cfg.annotation_path
        self.img_path = self.cfg.img_path
        self.input_size = self.cfg.input_size
        self.batch_size = self.cfg.batch_size
        self.aug_data = self.cfg.data_aug

        self.strides = self.yolo_cfg.strides
        self.classes = common.read_class_names(self.yolo_cfg.class_name)
        self.num_classes = len(self.classes)
        self.anchors = np.array(common.get_anchors(self.yolo_cfg.anchors))
        self.anchor_per_scale = self.yolo_cfg.anchor_per_scale
        self.max_bbox_per_scale = 150

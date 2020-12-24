# ================================================================
#
#   Editor      : Pycharm
#   File name   : yolov3
#   Author      : HuangWei
#   Created date: 2020-12-23 22:25
#   Email       : 446296992@qq.com
#   Description : YOLO 模型
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
import tensorflow as tf
from tensorflow.keras import models

from layers import *


class Yolo(models.Model):
    def __init__(self, **kwargs):
        super(Yolo, self).__init__(**kwargs)

    def call(self, inputs, training=None, mask=None):
        pass
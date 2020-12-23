# ================================================================
#
#   Editor      : Pycharm
#   File name   : train
#   Author      : HuangWei
#   Created date: 2020-12-23 14:51
#   Email       : 446296992@qq.com
#   Description : 开启训练
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================

import tensorflow as tf

from utils.config import YoloConfig as yolo_cfg
from utils.config import TrainConfig as train_cfg

# 动态分配显存
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


class Yolo:
    def __init__(self):
        self.global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)

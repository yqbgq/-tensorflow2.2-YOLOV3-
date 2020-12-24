# ================================================================
#
#   Editor      : Pycharm
#   File name   : test_yolov3
#   Author      : HuangWei
#   Created date: 2020-12-24 12:01
#   Email       : 446296992@qq.com
#   Description : 
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
from model import yolov3
import numpy as np
import tensorflow as tf

from model import backbone
from utils import dataloader
# 动态分配显存
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')

back = backbone.DarkNet53(True)

a = dataloader.Dataloader("train")

a = iter(a)

images, targets = next(a)

y = yolov3.Yolo(trainable=True)

result = back(images)

del back

print("now")
result = y(result)

print("hello!")

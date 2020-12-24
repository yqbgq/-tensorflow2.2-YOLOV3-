# ================================================================
#
#   Editor      : Pycharm
#   File name   : test_backbone
#   Author      : HuangWei
#   Created date: 2020-12-23 22:29
#   Email       : 446296992@qq.com
#   Description : 测试一下骨干网络的输出
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
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

back = backbone.DarkNet53()

a = dataloader.Dataloader("train")

a = iter(a)

images, targets = next(a)

result = np.array(back(images))

print("hello")

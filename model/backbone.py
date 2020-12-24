# ================================================================
#
#   Editor      : Pycharm
#   File name   : backbone
#   Author      : HuangWei
#   Created date: 2020-12-23 22:10
#   Email       : 446296992@qq.com
#   Description : 模型中的骨干网络
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
from tensorflow.keras import models

from model.layers import *


class DarkNet53(models.Model):
    def __init__(self):
        super(DarkNet53, self).__init__()
        self.__build_model()

    def __build_model(self):
        self.model1 = models.Sequential([
            conv([3, 3, 3, 32]),
            conv([3, 3, 32, 64], down=True),
            res_block(64, 32, 64),
            conv([3, 3, 64, 128], down=True),
            res_block(128, 64, 128),
            res_block(128, 64, 128),
            conv([3, 3, 128, 256], down=True),
            res_block(256, 128, 256), res_block(256, 128, 256), res_block(256, 128, 256),
            res_block(256, 128, 256), res_block(256, 128, 256), res_block(256, 128, 256),
            res_block(256, 128, 256), res_block(256, 128, 256)
        ])

        self.model2 = models.Sequential([
            conv([3, 3, 256, 512], down=True),
            res_block(512, 256, 512), res_block(512, 256, 512), res_block(512, 256, 512),
            res_block(512, 256, 512), res_block(512, 256, 512), res_block(512, 256, 512),
            res_block(512, 256, 512), res_block(512, 256, 512)
        ])

        self.model3 = models.Sequential([
            conv([3, 3, 512, 1024], down=True),
            res_block(1024, 512, 1024), res_block(1024, 512, 1024),
            res_block(1024, 512, 1024), res_block(1024, 512, 1024)
        ])

    def call(self, inputs, training=None, mask=None):
        route_1 = self.model1(inputs)
        route_2 = self.model2(route_1)
        route_3 = self.model3(route_2)

        return route_1, route_2, route_3

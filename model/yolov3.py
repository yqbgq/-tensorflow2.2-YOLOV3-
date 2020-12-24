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
from tensorflow.keras import models

from model.layers import *
from model import backbone
from utils import cal_statics
from utils.config import YoloConfig


class Yolo(models.Model):
    def __init__(self, trainable, **kwargs):
        super(Yolo, self).__init__(**kwargs)
        self.class_num = YoloConfig.classes_num
        self.trainable = trainable

        self.backbone = backbone.DarkNet53(trainable=self.trainable)

        self.__build_model()

    def call(self, inputs, training=None, mask=None):
        route_1, route_2, route_3 = self.backbone(inputs, training=self.trainable)
        # route_1, route_2, route_3 = inputs
        smaller_bbox, medium_bbox, larger_bbox = self.__forward([route_1, route_2, route_3])
        smaller_predict = cal_statics.decode(smaller_bbox, 0)
        medium_predict = cal_statics.decode(medium_bbox, 1)
        larger_predict = cal_statics.decode(larger_bbox, 2)
        return [[smaller_bbox, smaller_predict], [medium_bbox, medium_predict], [larger_bbox, larger_predict]]

    def __forward(self, routes):
        route_1, route_2, route_3 = routes

        larger_bbox = self.model1(route_3)

        temp_route = self.model2(larger_bbox)
        route_2 = tf.concat([temp_route, route_2], axis=-1)

        medium_bbox = self.model3(route_2)

        temp_route = self.model4(medium_bbox)
        route_1 = tf.concat([temp_route, route_1], axis=-1)

        smaller_bbox = self.model5(route_1)
        return [smaller_bbox, medium_bbox, larger_bbox]

    def __build_model(self):
        # 第一个卷积，获得最大分辨率
        self.model1 = models.Sequential([
            conv([1, 1, 1024, 512]),
            conv([3, 3, 512, 1024]),
            conv([1, 1, 1024, 512]),
            conv([3, 3, 512, 1024]),
            conv([1, 1, 1024, 512]),
            conv([3, 3, 512, 1024]),
            conv([1, 1, 1024, 3 * (self.class_num + 5)], activate=False, bn=False)
        ])

        # 最大分辨率在进行一次卷积，然后up_sample进入下一个分辨率的计算
        self.model2 = models.Sequential([
            conv([1, 1, 512, 256]),
            up_sample()
        ])

        # 获取中等分辨率
        self.model3 = models.Sequential([
            conv([1, 1, 768, 256]),
            conv([3, 3, 256, 512]),
            conv([1, 1, 512, 256]),
            conv([3, 3, 256, 512]),
            conv([1, 1, 512, 256]),
            conv([3, 3, 256, 512]),
            conv([1, 1, 512, 3 * (self.class_num + 5)], activate=False, bn=False),
        ])

        self.model4 = models.Sequential([
            conv([1, 1, 256, 128]),
            up_sample()
        ])

        self.model5 = models.Sequential([
            conv([1, 1, 384, 128]),
            conv([3, 3, 128, 256]),
            conv([1, 1, 256, 128]),
            conv([3, 3, 128, 256]),
            conv([1, 1, 256, 128]),
            conv([3, 3, 128, 256]),
            conv([1, 1, 256, 3 * (self.class_num + 5)], activate=False, bn=False),
        ])

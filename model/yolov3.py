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
from utils import cal_statics
from utils.config import YoloConfig


class Yolo(models.Model):
    def __init__(self, trainable, **kwargs):
        super(Yolo, self).__init__(**kwargs)
        self.class_num = YoloConfig.classes_num
        self.trainable = trainable

        self.__build_model()

    def save_model(self):
        """保存模型的参数"""
        block_list = [self, self.model1, self.model2, self.model3, self.model4, self.model5,
                      self.backbone_model1, self.backbone_model2, self.backbone_model3]
        for block in block_list:
            with open("../weights/{}".format(block.name), "wb+") as f:
                pickle.dump(block.get_weights(), f)
            for layer in block.layers:
                if "conv" in layer.name or "res" in layer.name:
                    layer.save_layer()

    def load_model(self):
        """加载模型的参数"""
        block_list = [self, self.model1, self.model2, self.model3, self.model4, self.model5,
                      self.backbone_model1, self.backbone_model2, self.backbone_model3]
        for block in block_list:
            with open("../weights/{}".format(block.name), "rb+") as f:
                weights = pickle.load(f)
                block.set_weights(weights)
            for layer in block.layers:
                if "conv" in layer.name or "res" in layer.name:
                    layer.load_layer()

    def get_config(self):
        return {"trainable": self.trainable, "class_num": self.class_num}

    def call(self, inputs, training=None, mask=None):
        """前向传播"""
        route_1 = self.backbone_model1(inputs)  # SHAPE: [N, 52, 52, 256]
        route_2 = self.backbone_model2(route_1)  # SHAPE: [N, 26, 26, 512]
        route_3 = self.backbone_model3(route_2)  # SHAPE: [N, 13, 13, 1024]

        """需要主义的是，这里 smaller_bbox 是小物体在大分辨率下小物体的预测框，而不是小分辨率下的检测结果"""
        smaller_bbox, medium_bbox, larger_bbox = self.__forward([route_1, route_2, route_3])

        """三种预测框经过解码之后得到的预测结果"""
        smaller_predict = cal_statics.decode(smaller_bbox, 0)
        medium_predict = cal_statics.decode(medium_bbox, 1)
        larger_predict = cal_statics.decode(larger_bbox, 2)
        return [[smaller_bbox, smaller_predict], [medium_bbox, medium_predict], [larger_bbox, larger_predict]]

    def __forward(self, routes):
        """获取routes是 backbone 的输出结果，在这里的 forward 是 YOLO 层的前向传播方式"""
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
        self.backbone_model1 = models.Sequential([
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

        self.backbone_model2 = models.Sequential([
            conv([3, 3, 256, 512], down=True),
            res_block(512, 256, 512), res_block(512, 256, 512), res_block(512, 256, 512),
            res_block(512, 256, 512), res_block(512, 256, 512), res_block(512, 256, 512),
            res_block(512, 256, 512), res_block(512, 256, 512)
        ])

        self.backbone_model3 = models.Sequential([
            conv([3, 3, 512, 1024], down=True),
            res_block(1024, 512, 1024), res_block(1024, 512, 1024),
            res_block(1024, 512, 1024), res_block(1024, 512, 1024)
        ])

        # 第一个卷积，获得最小分辨率
        self.model1 = models.Sequential([
            conv([1, 1, 1024, 512]),
            conv([3, 3, 512, 1024]),
            conv([1, 1, 1024, 512]),
            conv([3, 3, 512, 1024]),
            conv([1, 1, 1024, 512]),
            conv([3, 3, 512, 1024]),
            conv([1, 1, 1024, 3 * (self.class_num + 5)], activate=False, bn=False)
        ])

        # 最小分辨率在进行一次卷积，然后up_sample进入下一个分辨率的计算
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

        # 获取最大分辨率
        self.model5 = models.Sequential([
            conv([1, 1, 384, 128]),
            conv([3, 3, 128, 256]),
            conv([1, 1, 256, 128]),
            conv([3, 3, 128, 256]),
            conv([1, 1, 256, 128]),
            conv([3, 3, 128, 256]),
            conv([1, 1, 256, 3 * (self.class_num + 5)], activate=False, bn=False),
        ])

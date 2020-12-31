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

import cv2
import tensorflow as tf
import numpy as np

from utils import config
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

        self.strides = np.array(self.yolo_cfg.strides)
        self.classes = self.yolo_cfg.classes
        self.num_classes = self.yolo_cfg.classes_num
        self.anchors = np.array(self.yolo_cfg.anchors_list)
        self.anchor_per_scale = self.yolo_cfg.anchor_per_scale
        self.max_bbox_per_scale = 150

        self.annotations = common.load_annotations(self.annotation_path)
        self.sample_num = len(self.annotations)
        # 最后一个 batch 因为不足够，所以就舍去，因为每个epoch都会先将注释文件乱序
        # 所以所有图片都有可能被访问到，在读取注释文件的时候就已经乱序了
        self.batch_num = self.sample_num // self.batch_size
        self.batch_count = 0

        self.output_size = self.input_size // self.strides

    def __len__(self):
        return self.batch_num

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device('/cpu:0'):

            batch_image = np.zeros((self.batch_size, self.input_size, self.input_size, 3), dtype=np.float32)

            batch_label_smaller_bbox = np.zeros((self.batch_size, self.output_size[0], self.output_size[0],
                                                 self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_medium_bbox = np.zeros((self.batch_size, self.output_size[1], self.output_size[1],
                                                self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_larger_bbox = np.zeros((self.batch_size, self.output_size[2], self.output_size[2],
                                                self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)

            batch_smaller_bboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_medium_bboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_larger_bboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)

            num = 0
            # 当还可以从数据集中取出 batch 时，则取出，否则停止迭代
            if self.batch_count < self.batch_num:
                # 取出 batch_size 个图片和 label
                while num < self.batch_size:
                    # 计算图片的索引框架
                    index = self.batch_count * self.batch_size + num
                    annotation = self.annotations[index]
                    # 获取填充得到的图片和预处理之后的真值框
                    # 这里的 bboxes 是一张图片上所有真值框的集合 [N, 5]
                    image, bboxes = self.parse_annotation(annotation)

                    label_smaller_bbox, label_medium_bbox, label_larger_bbox, smaller_bboxes \
                        , medium_bboxes, larger_bboxes = self.preprocess_true_boxes(bboxes)

                    batch_image[num, :, :, :] = image
                    batch_label_smaller_bbox[num, :, :, :, :] = label_smaller_bbox
                    batch_label_medium_bbox[num, :, :, :, :] = label_medium_bbox
                    batch_label_larger_bbox[num, :, :, :, :] = label_larger_bbox
                    batch_smaller_bboxes[num, :, :] = smaller_bboxes
                    batch_medium_bboxes[num, :, :] = medium_bboxes
                    batch_larger_bboxes[num, :, :] = larger_bboxes
                    num += 1
                self.batch_count += 1
                batch_smaller_target = batch_label_smaller_bbox, batch_smaller_bboxes
                batch_medium_target = batch_label_medium_bbox, batch_medium_bboxes
                batch_larger_target = batch_label_larger_bbox, batch_larger_bboxes

                # 返回的是图片集合，小分辨率真值，中分辨率真值和大分辨率真值
                # 需要注意的是里面的坐标好像是原图上的坐标
                # TODO 在 label 和 bboxes 中，都存在坐标信息，似乎冗余
                # smaller意思是可以在最大分辨率下，寻找比较小的物体，larger则是在较低分辨率下找寻较大的物体
                return batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target)
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def parse_annotation(self, annotation):
        """
        解析注释文件，包括文件中的图片地址和标记框数据

        :param annotation: 注释信息
        :return:
        """
        line = annotation.split()
        image_path = line[0]

        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)

        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        bboxes = np.array([[int(i) for i in x.split(",")] for x in line[1:]])

        # 不进行数据增强，COCO2014 的数据已经很大了
        # if self.aug_data:
        #     image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
        #     image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
        #     image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        # 对图片进行预处理，预处理之后需要转换 bbox 的值
        image, bboxes = common.preprocess_img(np.copy(image), [self.input_size, self.input_size], np.copy(bboxes))
        return image, bboxes

    def preprocess_true_boxes(self, bboxes):
        """
        对真值框进行预处理

        :param bboxes: 真值框
        :return:
        """
        # 一张图片在三种不同分辨率下的标签，如 [52, 52, 3, 85]
        label = [np.zeros((self.output_size[i], self.output_size[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        # 三种不同分辨率下，真值框的坐标 [3, 150, 4]
        # 感觉在这里，不需要 150 种这么多，应该和 pytorch 版本中50个那么多就够了
        bboxes_x_y_w_h = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        # 三种分辨率下，真值框数量的计数
        bbox_count = np.zeros((3,))

        # 遍历每一个真值框
        for bbox in bboxes:
            # 真值框的坐标
            bbox_coordinate = bbox[:4]
            # 真值框中物体的类别
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0

            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)

            # 在这里，对 onehot 进行平滑？ 有利于计算损失嘛？
            delta = 0.01
            smooth_onehot = onehot * (1 - delta) + delta * uniform_distribution

            # 转换坐标系，本来的坐标是锚框的左上角坐标和右下角坐标
            # 现在变为锚框的中心点坐标和锚框的 W 和锚框的 H
            bbox_x_y_w_h = np.concatenate(
                [
                    (bbox_coordinate[2:] + bbox_coordinate[:2]) * 0.5,
                    bbox_coordinate[2:] - bbox_coordinate[:2]
                ], axis=-1
            )
            # 将真值框缩放到对应的缩放上
            bbox_x_y_w_h_scaled = 1.0 * bbox_x_y_w_h[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            # 第 i 种分辨率
            for i in range(3):
                # 处理锚框的 XY WH
                anchors_x_y_w_h = np.zeros((self.anchor_per_scale, 4))

                # TODO 这里又为什么这样处理锚框的坐标啊，迷惑
                anchors_x_y_w_h[:, 0:2] = np.floor(bbox_x_y_w_h_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_x_y_w_h[:, 2:4] = self.anchors[i]

                # 上面对缩放后的真值锚框进行了部分的位移，判断位移之后两个锚框之间 IOU
                # 感觉有点多余？
                iou_scale = common.bbox_iou(bbox_x_y_w_h_scaled[i][np.newaxis, :], anchors_x_y_w_h)
                iou.append(iou_scale)
                # TODO 可能是为了过滤比较小的目标嘛？
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_x_y_w_h_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    # TODO 放入的还是在原图上的坐标，但是转换为了锚框中间以及锚框的 W 和 H
                    # TODO bboxes_x_y_w_h 和 label 里面都有在原图上真值框的坐标，冗余嘛?
                    label[i][yind, xind, iou_mask, 0:4] = bbox_x_y_w_h
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    # 获取一个 bbox 的索引，如果超过了该分辨率上最大锚框数目，则循环覆盖
                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_x_y_w_h[i][bbox_ind, :4] = bbox_x_y_w_h
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_x_y_w_h_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_x_y_w_h
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_x_y_w_h[best_detect][bbox_ind, :4] = bbox_x_y_w_h
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_x_y_w_h
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

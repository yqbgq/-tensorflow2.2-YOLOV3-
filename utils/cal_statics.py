# ================================================================
#
#   Editor      : Pycharm
#   File name   : cal_statics
#   Author      : HuangWei
#   Created date: 2020-12-24 12:05
#   Email       : 446296992@qq.com
#   Description : 一些统计值的计算工具
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
import numpy as np
import tensorflow as tf

from utils.config import YoloConfig

classes_num = YoloConfig.classes_num
anchors = YoloConfig.anchors_list
strides = np.array(YoloConfig.strides)
iou_loss_threshold = YoloConfig.iou_loss_threshold


def decode(conv_output, i=0):
    """
    对预测的结果进行解码

    :param conv_output: 卷积的输出
    :param i: 表示分辨率的索引
    :return: 解码之后的张量 其形状为： [batch_siz, output_size, output_size, anchor_per_scale, 5 + num_classes]
             5 + num_classes 包含的内容为回归框的 x、y、w、h，回归框的置信度，各类的概率
    """

    conv_shape = tf.shape(conv_output)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]

    # conv_output 原本的形状 [batch_size, output_size, output_size, 3 * (5+classes_num)]
    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + classes_num))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]    # 获得 X 和 Y
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]    # 获得 W 和 H
    conv_raw_conf = conv_output[:, :, :, :, 4:5]    # 获得回归框的置信度
    conv_raw_prob = conv_output[:, :, :, :, 5:]     # 获得各类的概率

    # 这里 x 和 y 的形状是 [output_size, output_size]，如[52, 52]
    # 代表的是各个分块
    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

    # xy_grid 的 shape 为 [batch_num, output_size, output_size, 3, 2]
    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    # 对 XY 和 WH 及逆行解码
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * strides[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * anchors[i]) * strides[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    # 解码置信度和各类的概率
    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def bbox_iou(boxes1, boxes2):
    """计算两个框的交并比"""
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    # 原本是中心点坐标加上 W 和 H，现在这里转换成左上角坐标和右下角坐标
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area


def bbox_giou(boxes1, boxes2):
    """
    计算两个框的 GIOU

    GIOU 是一种对 IOU 的改进， IOU 主要存在这的问题是当两个框没有重叠时， 无论两个框相距多远，IOU 都为0

    GIOU 的计算方式：计算两个矩形框的最小外包矩形，求出最小外包矩形中不属于两个框的面积大小占整个外包矩形的比例
                   计算 IOU ， GIOU = IOU - 比例

    故 GIOU 的取值范围是 (-1, 1]

    """
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


def compute_loss(pred, conv, label, bboxes, i=0):
    """
    计算损失，分别是 GIOU 损失， 置信度损失以及 分类概率损失

    :param pred: 卷积输出的结果 conv 经过解码之后得到的结果
    :param conv: 卷积输出的结果
    :param label: 真值 5 + classes_num， 输出结果通过 sigmoid 计算得到的
    :param bboxes: 真值 框的 x_y_w_h
    :param i: 分辨率的索引
    """
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = strides[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + classes_num))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]   # 表示回归框的置信度
    label_prob = label[:, :, :, :, 5:]      # 表示每种类别的概率

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    # 计算 GIOU 的损失
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    # 对于每一个预测框，预测失误的 mask 矩阵
    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < iou_loss_threshold, tf.float32)

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    # 计算置信度损失函数
    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    # 计算类别预测损失函数
    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return giou_loss, conf_loss, prob_loss

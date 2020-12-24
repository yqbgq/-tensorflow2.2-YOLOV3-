# ================================================================
#
#   Editor      : Pycharm
#   File name   : common
#   Author      : HuangWei
#   Created date: 2020-12-23 15:10
#   Email       : 446296992@qq.com
#   Description : 公共工具
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
import cv2
import numpy as np


def read_class_names(class_file_name):
    """loads class name from a file"""
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def get_anchors(anchor_path):
    """loads the anchors from a file"""
    with open(anchor_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape([3, 3, 2])


def load_annotations(path):
    """读取注释信息"""
    with open(path, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
    # np.random.shuffle(annotations)
    return annotations


def preprocess_img(image, target_size, gt_boxes=None):
    """对图片进行预处理"""
    target_h, target_w = target_size
    image_h, image_w, _ = image.shape

    # 获取缩放的倍数
    scale = min(target_w / image_w, target_h / image_h)
    # 获取改变后的高和宽
    new_w, new_h = int(scale * image_w), int(scale * image_h)
    image_resized = cv2.resize(image, (new_w, new_h))

    # 填充模板
    padded_image = np.full(shape=[target_h, target_w, 3], fill_value=128.0)
    dw, dh = (target_w - new_w) // 2, (target_h - new_h) // 2

    # 将 resized 之后图片放在填充模板的中间
    padded_image[dh:new_h + dh, dw:new_w + dw, :] = image_resized
    # 将图片的内容转换到 0-1
    padded_image = padded_image / 255.

    if gt_boxes is None:
        return padded_image
    else:
        # 将真值框转换到缩放后的图片上
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return padded_image, gt_boxes


def bbox_iou(boxes1, boxes2):
    """计算两个 boxes的IOU """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return inter_area / union_area

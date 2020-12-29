# ================================================================
#
#   Editor      : Pycharm
#   File name   : detect
#   Author      : HuangWei
#   Created date: 2020-12-29 10:37
#   Email       : 446296992@qq.com
#   Description : 使用已经训练好的权重文件来进行检测
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
import cv2
import tensorflow as tf
import numpy as np

from model import yolov3
from utils import common

from PIL import Image


def detect(img_path):
    input_size = 416
    model = yolov3.Yolo(trainable=False)

    model.summary()

    original_image = cv2.imread(img_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    image_data = common.preprocess_img(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    pred_bbox = model.predict(image_data)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = common.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
    bboxes = common.nms(bboxes, 0.45, method='nms')

    image = common.draw_bbox(original_image, bboxes)
    image = Image.fromarray(image)
    image.show()



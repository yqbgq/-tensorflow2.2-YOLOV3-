# ================================================================
#
#   Editor      : Pycharm
#   File name   : detect
#   Author      : HuangWei
#   Created date: 2020-12-29 10:37
#   Email       : 446296992@qq.com
#   Description : 使用已经训练好的权重文件来进行检测测试
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
import cv2
import tensorflow as tf
import numpy as np
import os

from model import yolov3
from utils import common

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 表示使用 CPU 进行计算


def detect(img_path):
    input_size = 416
    model = yolov3.Yolo(trainable=False)

    # 读取测试数据
    original_image = cv2.imread(img_path)
    # 因为 CV2 读取图片是 BGR 通道顺序，所以要转换为 RGB 通道
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # 获取图像形状的前两个数字，是高和宽
    original_image_size = original_image.shape[:2]
    # 对图片进行预处理，将图片变成 input_size * input_size 大小
    image_data = common.preprocess_img(np.copy(original_image), [input_size, input_size])
    # 为图像矩阵添加一个新的维度，变成 batch_size(1) * input_size * input_size * channel_num
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    # 先进行一次前向传播，初始化模型的参数
    model(image_data)
    # 加载模型的已经训练好的参数
    model.load_model()

    # 获取预测结果
    pred_bbox = model(image_data)
    # 每一个预测结果为 classes_num + x_y_w_h + conf
    pred_bbox = [tf.reshape(x[1], (-1, tf.shape(x[1])[-1])) for x in pred_bbox]
    # 将三种分辨率下的框全部拼接在一起，形成一个张量
    pred_bbox = tf.concat(pred_bbox, axis=0)

    # 对候选框进行后处理，筛去一部分置信度太低的框
    bboxes = common.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.45)
    # 使用 NMS 方法，去除掉一些候选框，得到最佳的候选框
    bboxes = common.nms(bboxes, 0.45, method='nms')

    # 将候选框绘制在图上，左上角是类别，这里 classes 取 0-10 是因为使用的是数字进行识别
    image = common.draw_bbox(original_image, bboxes, classes=[i for i in range(10)])

    # 展示图像
    # cv2.imshow("result", image)
    # cv2.waitKey()
    cv2.imwrite("D://result.jpg", image)


if __name__ == "__main__":
    path = "../data/demo/3.jpg"
    detect(path)

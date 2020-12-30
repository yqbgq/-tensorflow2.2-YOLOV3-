# ================================================================
#
#   Editor      : Pycharm
#   File name   : config
#   Author      : HuangWei
#   Created date: 2020-12-23 14:16
#   Email       : 446296992@qq.com
#   Description : 项目的配置文件
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
from utils import common


class YoloConfig:
    class_name = "../data/classes/yymnist.names"
    anchors = "../data/anchors/baseline_anchors.txt"
    log_dir = "../log"

    strides = [8, 16, 32]
    anchor_per_scale = 3
    iou_loss_threshold = 0.5

    classes = common.read_class_names(class_name)
    classes_num = len(classes)
    anchors_list = common.get_anchors(anchors)


class TrainConfig:
    # 标签文件或者注释文件路径
    annotation_path = "../data/dataset/yymnist_train.txt"
    img_path = "../data/dataset/train"
    batch_size = 1
    input_size = 416
    # data_aug = True
    lr_init = 1e-3
    lr_end = 1e-6
    warmup_epochs = 2
    total_epochs = 30


class TestConfig:
    annotation_path = "../data/dataset/yymnist_test.txt"
    img_path = "../data/dataset/test"
    batch_size = 2
    input_size = 544
    # data_aug = False
    detected_image_path = "../data/detection/"
    score_threshold = 0.3
    iou_threshold = 0.45

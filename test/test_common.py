# ================================================================
#
#   Editor      : Pycharm
#   File name   : test_common
#   Author      : HuangWei
#   Created date: 2020-12-23 15:12
#   Email       : 446296992@qq.com
#   Description : 测试公共共具类
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
from utils import common


def test_read_class_names(path="../data/classes/coco.names"):
    names = common.read_class_names(path)
    print(len(names))


def test_get_anchors(anchor_path="../data/anchors/baseline_anchors.txt"):
    result = common.get_anchors(anchor_path)
    print(result)


def test_load_annotations(path="../data/dataset/yymnist_train.txt"):
    result = common.load_annotations(path)
    print(result[0])


if __name__ == "__main__":
    test_load_annotations()

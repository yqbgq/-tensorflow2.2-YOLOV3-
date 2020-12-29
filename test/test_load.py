# ================================================================
#
#   Editor      : Pycharm
#   File name   : test_load
#   Author      : HuangWei
#   Created date: 2020-12-29 11:17
#   Email       : 446296992@qq.com
#   Description : 
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
from model import yolov3

model = yolov3.Yolo(trainable=True)

model.load_weights("C://Users//huangwei//Desktop//fsdownload//yolov3_29.ckpt.index")
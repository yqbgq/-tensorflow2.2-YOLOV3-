# ================================================================
#
#   Editor      : Pycharm
#   File name   : test_dataloader
#   Author      : HuangWei
#   Created date: 2020-12-23 20:37
#   Email       : 446296992@qq.com
#   Description : 
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
from utils import dataloader

a = dataloader.Dataloader("train")

a = iter(a)

c = next(a)


print("hello")
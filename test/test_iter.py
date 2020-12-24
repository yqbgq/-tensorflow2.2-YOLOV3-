# ================================================================
#
#   Editor      : Pycharm
#   File name   : test_iter
#   Author      : HuangWei
#   Created date: 2020-12-23 19:05
#   Email       : 446296992@qq.com
#   Description : 
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
class test:
    def __iter__(self):
        return self

    def __init__(self):
        self.count = 0
        self.count_num = 3

    def __next__(self):
        if self.count < self.count_num:
            self.count += 1
            return self.count
        else:
            self.count = 0
            raise StopIteration


a = test()
a = iter(a)
print(next(a))
print(next(a))
print(next(a))
print(next(a))
print(next(a))

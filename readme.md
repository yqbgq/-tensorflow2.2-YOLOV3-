# 基于Tensorflow2.2的YOLOv3实现

## 前言

本实现的目标是实现 YOLOV3 的网络，学习使用神经网络框架搭建自定义神经网络

**本仓库只完成了最轻量化的实现，使用 yymnist 快速进行训练和测试**

项目进行了比较详细的中文注释，但是也局限于自身的能力，难免会出现错误，大家自行甄别

## 致谢

该实现参考了以下大佬的代码实现以及十分有用的博客，感谢大佬的贡献

- **bobo0810(github)**  [Github仓库地址](https://github.com/bobo0810/PytorchNetHub/tree/master/Yolov3_pytorch)
- **YunYang1994(github)** [Github仓库地址](https://github.com/yqbgq/TensorFlow2.0-Examples/tree/f99fcef22caa2758b5eefce10ee789384345506d/4-Object_Detection/YOLOV3)
- **太阳花的小绿豆(CSDN)** [YOLO v3网络结构分析-博客地址](https://blog.csdn.net/qq_37541097/article/details/81214953)
- **江大白(CSDN)*** [Yolov3可视化网络结构图-博客地址](https://blog.csdn.net/nan355655600/article/details/106246355/)

尤其感谢 **YunYang1994(github)** ，本仓库使用了大佬的 yymnist 数据集创建程序，并且基本上便是按照大佬的 YOLOV3 实现进行了适合自己理解的些许改动

## 说明


本来写了一个用 Pytorch 的实现的，但是在训练过程中发现内存或者显存出现不断增长，而且不收敛或者收敛缓慢的情况

考虑了很多情况也没有好转，同时也烦恼于 Pytorch 手动转换张量位置以及是否需要计算张量的麻烦，所以决定还是用 TF 写

本来因为自己电脑上 TF 一直报错卷积算法有问题，后来降级到2.2发现欸，可以了

**主要改动：**
1. 将网络的 backbone 和 YOLO 判别部分合并成为一个 yolov3 类
2. 将原仓库中的函数式模型改成了更加容易理解的序贯式模型，封装成一个类，使用 call 函数来控制前向传播，方便进行调试
3. 由于自定义层和自定义模型的原因，使用 TF 自带的保存权重方式我自己测试的时候无法成功。故而为每个层和模型分别写了保存和加载权重的函数，直接调用 model.save_model() 和 model.load_model() 即可保存和加载模型
4. 将配置方法改成了配置类

## 安装

**安装项目依赖**

```
pip install -r requirements.txt
```

**获取网络权重**

百度网盘
```angular2html
链接: https://pan.baidu.com/s/1uTzEE3I_83Z3JJva_8FcxA 
提取码: 3cg7
```

谷歌云硬盘
```angular2html
https://drive.google.com/file/d/1yswFfenc7oewre0q6JkBkLRGyTnoKjSI/view?usp=sharing
```

## 结果展示

**demo展示**

<p align="center">
    <img width="100%" src="https://github.com/yqbgq/-tensorflow2.2-YOLOV3-/blob/master/data/demo/demo.png" style="max-width:100%;">
    </a>
</p>

**检测结果**
<p align="center">
    <img width="100%" src="https://github.com/yqbgq/-tensorflow2.2-YOLOV3-/blob/master/data/demo/result.png" style="max-width:100%;">
    </a>
</p>

## 使用

**克隆 [yymnist](https://github.com/YunYang1994/yymnist) 并创建数据**

```
git clone https://github.com/YunYang1994/yymnist.git
python yymnist/make_data.py --images_num 10000 --images_path ./data/dataset/train --labels_txt ./data/dataset/yymnist_train.txt
python yymnist/make_data.py --images_num 200  --images_path ./data/dataset/test  --labels_txt ./data/dataset/yymnist_test.txt
```

将得到的 train 和 test 数据集和标签文件放置在 /data/dataset 下

**data文件夹组织方式**
![](https://amos-blog.oss-accelerate.aliyuncs.com/img/21/1/dataset组织方式.png)
<p align="center">
    <img width="40%" src="https://amos-blog.oss-accelerate.aliyuncs.com/img/21/1/dataset组织方式.png" style="max-width:100%;">
    </a>
</p>

**使用如下代码来训练模型以及损失和学习率追踪**
```
cd ./script
python train.py
tensorboard --logdir ./data/log
```

**训练完之后使用如下代码来测试模型效果**
```
cd ./script
python detect.py
```

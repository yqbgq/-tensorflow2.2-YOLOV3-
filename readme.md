# 基于Tensorflow2.2的YOLOv3实现

---
该实现参考了以下大佬的代码实现以及十分有用的博客，感谢大佬的贡献

- **bobo0810(github)**  [Github仓库地址](https://github.com/bobo0810/PytorchNetHub/tree/master/Yolov3_pytorch)
- **YunYang1994(github)** [Github仓库地址](https://github.com/yqbgq/TensorFlow2.0-Examples/tree/f99fcef22caa2758b5eefce10ee789384345506d/4-Object_Detection/YOLOV3)
- **太阳花的小绿豆(CSDN)** [YOLO v3网络结构分析-博客地址](https://blog.csdn.net/qq_37541097/article/details/81214953)
- **江大白(CSDN)*** [Yolov3可视化网络结构图-博客地址](https://blog.csdn.net/nan355655600/article/details/106246355/)
---

本来写了一个用 Pytorch 的实现的，但是在训练过程中发现内存或者显存出现不断增长，而且不收敛或者收敛缓慢的情况

考虑了很多情况也没有好转，同时也烦恼于 Pytorch 手动转换张量位置以及是否需要计算张量的麻烦，所以决定还是用 TF 写

本来因为自己电脑上 TF 一直报错卷积算法有问题，后来降级到2.2发现欸，可以了

# 环境

| python版本  |  TF版本 |
| ----------- | ----------   |
|  3.8  | 2.2   |

# Waiting for completion
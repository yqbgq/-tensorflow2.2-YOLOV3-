# ================================================================
#
#   Editor      : Pycharm
#   File name   : layers
#   Author      : HuangWei
#   Created date: 2020-12-23 21:34
#   Email       : 446296992@qq.com
#   Description : 模型中通用的层
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
import pickle

import tensorflow as tf
from tensorflow.keras import layers


class BatchNormalization(tf.keras.layers.BatchNormalization):
    # BatchNormalization 设置 training 为 False，可以让其使用全局数据计算来归一化
    # 而不是使用批次数据来计算归一化
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


class conv(layers.Layer):
    def __init__(self, filters_shape, down=False, activate=True, bn=True, **kwargs):
        """
        自定义卷积层

        :param filters_shape: 卷积核形状
        :param down: 是否是向下采样，即是否保持 feature map 的大小一致
        :param activate: 是否使用激活函数
        :param bn: 是否使用 BatchNormalization
        :param kwargs: 其他参数
        """
        super(conv, self).__init__(**kwargs)
        self.filters_shape = filters_shape
        self.down = down
        self.activate = activate
        self.bn = bn

        self.__build_layer()

    def __build_layer(self):
        if self.down:
            self.padding = 'valid'
            self.strides = 2
        else:
            self.strides = 1
            self.padding = 'same'

        # 构造卷积核
        self.conv = tf.keras.layers.Conv2D(filters=self.filters_shape[-1], kernel_size=self.filters_shape[0],
                                           strides=self.strides, padding=self.padding, use_bias=not self.bn,
                                           kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                           bias_initializer=tf.constant_initializer(0.))

    # 定义前向传播函数
    def call(self, inputs, **kwargs):
        if self.down:
            inputs = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)
        output = self.conv(inputs)
        if self.bn:
            output = BatchNormalization()(output)
        if self.activate:
            output = tf.nn.leaky_relu(output, alpha=0.1)
        return output

    def save_layer(self):
        # 保存层的权重
        with open("../weights/{}".format(self.name), "wb+") as f:
            pickle.dump(self.get_weights(), f)

    def load_layer(self):
        # 加载层的权重
        with open("../weights/{}".format(self.name), "rb+") as f:
            weights = pickle.load(f)
            self.set_weights(weights)


class res_block(layers.Layer):
    def __init__(self, input_channel, filter_num1, filter_num2, **kwargs):
        """
        初始化自定义 res_block
        :param input_channel: 输入的通道数
        :param filter_num1: 第一个卷积核的输出通道数目
        :param filter_num2: 第二个卷积核的输出通道数目
        :param kwargs: 其他参数
        """
        super(res_block, self).__init__(**kwargs)
        self.input_channel = input_channel
        self.filter_num1 = filter_num1
        self.filter_num2 = filter_num2
        self.__build_layer()

    def __build_layer(self):
        self.conv1 = conv(filters_shape=(1, 1, self.input_channel, self.filter_num1))
        self.conv2 = conv(filters_shape=(3, 3, self.filter_num1, self.filter_num2))

    def call(self, inputs, **kwargs):
        """前向传播"""
        short_cut = inputs
        output = self.conv1(inputs)
        output = self.conv2(output)
        output = short_cut + output
        return output

    def save_layer(self):
        """保存层的权重，并且调用内里的两个卷积层的保存权重方法"""
        with open("../weights/{}".format(self.name), "wb+") as f:
            pickle.dump(self.get_weights(), f)
        self.conv1.save_layer()
        self.conv2.save_layer()

    def load_layer(self):
        """加载层和内里两个卷积层的权重"""
        with open("../weights/{}".format(self.name), "rb+") as f:
            weights = pickle.load(f)
            self.set_weights(weights)
        self.conv1.load_layer()
        self.conv2.load_layer()


class up_sample(layers.Layer):
    def __init__(self, **kwargs):
        super(up_sample, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """将图像放大两倍"""
        return tf.image.resize(inputs, (inputs.shape[1] * 2, inputs.shape[2] * 2), method='nearest')


# class BatchNormalization(tf.keras.layers.BatchNormalization):
#     """
#     "Frozen state" and "inference mode" are two separate concepts.
#     `layer.trainable = False` is to freeze the layer, so the layer will use
#     stored moving `var` and `mean` in the "inference mode", and both `gama`
#     and `beta` will not be updated !
#     """
#
#     def call(self, x, training=False):
#         if not training:
#             training = tf.constant(False)
#         training = tf.logical_and(training, self.trainable)
#         return super().call(x, training)
#
#
# class conv(layers.Layer):
#     def __init__(self, filters_shape, down=False, activate=True, bn=True, **kwargs):
#         super(conv, self).__init__(**kwargs)
#         self.filters_shape = filters_shape
#         self.down = down
#         self.activate = activate
#         self.bn = bn
#
#         self.__build_layer()
#
#     def save(self):
#         with open("../weights/{}".format(self.name), "wb+") as f:
#             pickle.dump(self.weights, f)
#
#     def load(self):
#         with open("../weights/{}".format(self.name), "rb+") as f:
#             weights = pickle.load(f)
#             self.set_weights(weights)
#
#     def get_config(self):
#         return {"filters_shape": self.filters_shape, "down": self.down,
#                 "activate": self.activate, "bn": self.bn}
#
#     def __build_layer(self):
#         if self.down:
#             self.padding = 'valid'
#             self.strides = 2
#         else:
#             self.strides = 1
#             self.padding = 'same'
#         self.conv = tf.keras.layers.Conv2D(filters=self.filters_shape[-1], kernel_size=self.filters_shape[0],
#                                            strides=self.strides, padding=self.padding, use_bias=not self.bn,
#                                            kernel_regularizer=tf.keras.regularizers.l2(0.0005),
#                                            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
#                                            bias_initializer=tf.constant_initializer(0.))
#
#     def call(self, inputs, **kwargs):
#         if self.down:
#             inputs = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)
#         output = self.conv(inputs)
#         if self.bn:
#             output = BatchNormalization()(output)
#         if self.activate:
#             output = tf.nn.leaky_relu(output, alpha=0.1)
#         return output
#
#
# class res_block(layers.Layer):
#     def __init__(self, input_channel, filter_num1, filter_num2, **kwargs):
#         super(res_block, self).__init__(**kwargs)
#         self.input_channel = input_channel
#         self.filter_num1 = filter_num1
#         self.filter_num2 = filter_num2
#         self.__build_layer()
#
#     def save(self):
#         with open("../weights/{}".format(self.name), "wb+") as f:
#             pickle.dump(self.weights, f)
#         self.conv1.save()
#         self.conv2.save()
#
#     def load(self):
#         with open("../weights/{}".format(self.name), "rb+") as f:
#             weights = pickle.load(f)
#             self.set_weights(weights)
#         self.conv1.load()
#         self.conv2.load()
#
#     def get_config(self):
#         return {"input_channel": self.input_channel, "filter_num1": self.filter_num1,
#                 "filter_num2": self.filter_num2}
#
#     def __build_layer(self):
#         self.conv1 = conv(filters_shape=(1, 1, self.input_channel, self.filter_num1))
#         self.conv2 = conv(filters_shape=(3, 3, self.filter_num1, self.filter_num2))
#
#     def call(self, inputs, **kwargs):
#         short_cut = inputs
#         output = self.conv1(inputs)
#         output = self.conv2(output)
#         output = short_cut + output
#         return output
#
#
# class up_sample(layers.Layer):
#     def __init__(self, **kwargs):
#         super(up_sample, self).__init__(**kwargs)
#
#     def call(self, inputs, **kwargs):
#         return tf.image.resize(inputs, (inputs.shape[1] * 2, inputs.shape[2] * 2), method='nearest')
#
#

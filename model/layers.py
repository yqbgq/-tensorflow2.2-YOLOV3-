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
import tensorflow as tf
from tensorflow.keras import layers


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """

    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


class conv(layers.Layer):
    def __init__(self, filters_shape, down=False, activate=True, bn=True, **kwargs):
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
        self.conv = tf.keras.layers.Conv2D(filters=self.filters_shape[-1], kernel_size=self.filters_shape[0],
                                           strides=self.strides, padding=self.padding, use_bias=not self.bn,
                                           kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                           bias_initializer=tf.constant_initializer(0.))

    def call(self, inputs, **kwargs):
        if self.down:
            inputs = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)
        output = self.conv(inputs)
        if self.bn:
            output = BatchNormalization()(output)
        if self.activate:
            output = tf.nn.leaky_relu(output, alpha=0.1)
        return output


class res_block(layers.Layer):
    def __init__(self, input_channel, filter_num1, filter_num2, **kwargs):
        super(res_block, self).__init__(**kwargs)
        self.input_channel = input_channel
        self.filter_num1 = filter_num1
        self.filter_num2 = filter_num2
        self.__build_layer()

    def __build_layer(self):
        self.conv1 = conv(filters_shape=(1, 1, self.input_channel, self.filter_num1))
        self.conv2 = conv(filters_shape=(3, 3, self.filter_num1, self.filter_num2))

    def call(self, inputs, **kwargs):
        short_cut = inputs
        output = self.conv1(inputs)
        output = self.conv2(output)
        output = short_cut + output
        return output


class up_sample(layers.Layer):
    def __init__(self, **kwargs):
        super(up_sample, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return tf.image.resize(inputs, (inputs.shape[1] * 2, inputs.shape[2] * 2), method='nearest')


# def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True):
#     if downsample:
#         input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
#         padding = 'valid'
#         strides = 2
#     else:
#         strides = 1
#         padding = 'same'
#
#     conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size=filters_shape[0], strides=strides,
#                                   padding=padding,
#                                   use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
#                                   kernel_initializer=tf.random_normal_initializer(stddev=0.01),
#                                   bias_initializer=tf.constant_initializer(0.))(input_layer)
#
#     if bn:
#         conv = BatchNormalization()(conv)
#     if activate == True:
#         conv = tf.nn.leaky_relu(conv, alpha=0.1)
#
#     return conv
#
#
# def residual_block(input_layer, input_channel, filter_num1, filter_num2):
#     short_cut = input_layer
#     conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1))
#     conv = convolutional(conv, filters_shape=(3, 3, filter_num1, filter_num2))
#
#     residual_output = short_cut + conv
#     return residual_output
#
#
# def upsample(input_layer):
#     return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')

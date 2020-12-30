# ================================================================
#
#   Editor      : Pycharm
#   File name   : train
#   Author      : HuangWei
#   Created date: 2020-12-23 14:51
#   Email       : 446296992@qq.com
#   Description : 开启训练
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
import os
import shutil
import tensorflow as tf
from tqdm import trange
import numpy as np
import time
import sys

sys.path.append("..")   # 在云端服务器上，不加这个找不到相对导入的 utils 和 model

from utils import cal_statics
from utils.config import TrainConfig, YoloConfig
from utils.dataloader import Dataloader
from model import yolov3

# 动态分配显存，如果机器的显存比较小，使用这个可以避免一次性申请太多显存导致 OOM
# 当然，也不要指望能够有多大的作用
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


def main():
    log_dir = YoloConfig.log_dir                        # 获取输出日志的目录

    model = yolov3.Yolo(trainable=True)                 # 生成 YOLO 模型，设置 trainable 为 True

    optimizer = tf.keras.optimizers.Adam()              # 使用 Adam 优化器，后面会随着 epoch 更新学习率

    if os.path.exists(log_dir):                         # 检查输出日志的目录，清空目录下的过往日志
        shutil.rmtree(log_dir)

    writer = tf.summary.create_file_writer(log_dir)     # 生成一个 TensorBoard 的输出器

    global_steps = 1                                    # 全局步数，用于计算学习率

    for epoch in range(TrainConfig.total_epochs):       # 开始训练的迭代

        train_dataset = Dataloader('train')             # 通过数据集加载器，加载训练数据集
        steps_per_epoch = len(train_dataset)            # 计算该 epoch 内的步数

        # 计算 warm up 需要的步数和全局需要的步数
        warmup_steps = TrainConfig.warmup_epochs * steps_per_epoch
        total_steps = TrainConfig.total_epochs * steps_per_epoch

        with trange(len(train_dataset)) as t:           # 使用 tqdm 来显示输出条，同时可以修改描述和后缀描述
            # 计算各种损失的平均值，用于在后缀描述中的展示
            avg_giou_loss, avg_conf_loss, avg_prob_loss, avg_total_loss = 0.0, 0.0, 0.0, 0.0

            for step in t:
                # 获取图像数据和标签数据
                image_data, target = next(train_dataset)

                # 计算梯度
                with tf.GradientTape() as tape:
                    pred_result = model(image_data)             # 前向传播，得到预测的输出结果

                    giou_loss = conf_loss = prob_loss = 0       # 三种损失

                    # 对于三种分辨率下的预测框，计算相应的损失
                    for i in range(3):
                        conv, pred = pred_result[i][0], pred_result[i][1]
                        loss_items = cal_statics.compute_loss(pred, conv, *target[i], i)
                        giou_loss += loss_items[0]
                        conf_loss += loss_items[1]
                        prob_loss += loss_items[2]

                    # 计算总损失，使用总损失进行梯度下降
                    total_loss = giou_loss + conf_loss + prob_loss

                    # 计算平均损失，用于输出
                    avg_giou_loss = avg_giou_loss + giou_loss
                    avg_conf_loss = avg_conf_loss + conf_loss
                    avg_prob_loss = avg_prob_loss + prob_loss
                    avg_total_loss = avg_total_loss + total_loss

                    # 拼接描述和后缀，在滚动条进行显示
                    des = time.strftime('%m-%d %H:%M:%S', time.localtime(time.time())) + " Epoch {}".format(epoch)
                    post = "lr: {:.6f} giou_loss: {:.2f} conf_loss: {:.2f} prob_loss: {:.2f} total_loss: {:.2f}".format(
                        optimizer.lr.numpy(), avg_giou_loss / (step + 1),
                        avg_conf_loss / (step + 1), avg_prob_loss / (step + 1), avg_total_loss / (step + 1))

                    # 设置描述和后缀
                    t.set_description(des)
                    t.set_postfix_str(post)

                    # 计算梯度，进行梯度下降优化
                    gradients = tape.gradient(total_loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                    global_steps += 1   # 递增全局总步数

                    # 更新学习率，并且应用于优化器
                    if global_steps < warmup_steps:
                        lr = global_steps / warmup_steps * TrainConfig.lr_init
                    else:
                        lr = TrainConfig.lr_end + 0.4 * (TrainConfig.lr_init - TrainConfig.lr_end) * (
                            (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                        )
                    optimizer.lr.assign(lr)

                    # 将各种数值输出到可视化中
                    with writer.as_default():
                        tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                        tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                        tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                        tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                        tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
                    writer.flush()

        # 调用自定义函数，保存网络权重
        model.save_model()


if __name__ == "__main__":
    main()

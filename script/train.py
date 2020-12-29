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
import time

import tensorflow as tf
from tqdm import trange
import numpy as np

from utils import cal_statics
from utils.config import TrainConfig
from utils.config import YoloConfig
from utils.dataloader import Dataloader
from model import yolov3

# 动态分配显存
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


def main():
    log_dir = "../log"
    ckpt_patt = YoloConfig.ckpt_patt

    model = yolov3.Yolo(trainable=True)
    optimizer = tf.keras.optimizers.Adam()
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = tf.summary.create_file_writer(log_dir)

    global_steps = 1

    for epoch in range(TrainConfig.total_epochs):

        train_dataset = Dataloader('train')
        steps_per_epoch = len(train_dataset)
        warmup_steps = TrainConfig.warmup_epochs * steps_per_epoch
        total_steps = TrainConfig.total_epochs * steps_per_epoch

        with trange(len(train_dataset)) as t:
            for _ in t:
                image_data, target = next(train_dataset)
                with tf.GradientTape() as tape:
                    pred_result = model(image_data)
                    giou_loss = conf_loss = prob_loss = 0

                    for i in range(3):
                        conv, pred = pred_result[i][0], pred_result[i][1]
                        loss_items = cal_statics.compute_loss(pred, conv, *target[i], i)
                        giou_loss += loss_items[0]
                        conf_loss += loss_items[1]
                        prob_loss += loss_items[2]
                    total_loss = giou_loss + conf_loss + prob_loss

                    des = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + " Epoch {}".format(epoch)
                    post = "lr: {:.6f} giou_loss: {:.2f} conf_loss: {:.2f} prob_loss: {:.2f} total_loss: {:.2f}".format(
                        optimizer.lr.numpy(), giou_loss, conf_loss, prob_loss, total_loss)

                    t.set_description(des)
                    t.set_postfix_str(post)

                    gradients = tape.gradient(total_loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                    global_steps += 1

                    if global_steps < warmup_steps:
                        lr = global_steps / warmup_steps * TrainConfig.lr_init
                    else:
                        lr = TrainConfig.lr_end + 0.5 * (TrainConfig.lr_init - TrainConfig.lr_end) * (
                            (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                        )
                    optimizer.lr.assign(lr)

                    # writing summary data
                    with writer.as_default():
                        tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                        tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                        tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                        tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                        tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
                    writer.flush()
        if (epoch + 1) % 2 == 0:
            model.save_weights(ckpt_patt.format(epoch))


if __name__ == "__main__":
    main()

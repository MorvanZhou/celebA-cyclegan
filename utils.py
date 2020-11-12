import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys


def set_soft_gpu(soft_gpu):
    if soft_gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


def save_gan(model, ep):
    model_name = model.__class__.__name__.lower()
    r = np.random.choice([0, 1], (10, model.label_dim), replace=True)
    imgs = np.concatenate(
        [model.predict(r) for _ in range(5)],
        axis=0)

    imgs = (imgs + 1) / 2
    plt.clf()
    nc, nr = 10, 5
    plt.figure(0, (nc * 2, nr * 2))
    for c in range(nc):
        for r in range(nr):
            i = r * nc + c
            plt.subplot(nr, nc, i + 1)
            plt.imshow(imgs[i])
            plt.axis("off")

    plt.tight_layout()
    path = "visual/{}/{}.png".format(model_name, ep)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)


def get_logger(model_name, date_str):
    log_fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = "visual/{}/{}/train.log".format(model_name, date_str)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(log_fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(log_fmt)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    return logger
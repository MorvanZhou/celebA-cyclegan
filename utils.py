import tensorflow as tf
import os
import matplotlib.pyplot as plt
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


def save_gan(model, img_name, img_women, img_men):
    img_women = img_women
    img_men = img_men
    img_women_ = model.g12.predict(img_men)  # man to woman
    img_men_ = model.g21.predict(img_women)  # woman to man

    convt = lambda x: (x + 1) / 2
    imgs = [convt(img_women), convt(img_men_), convt(img_men), convt(img_women_)]
    plt.clf()
    nc, nr = len(img_women), 4
    plt.figure(0, (nc*2, nr*2))
    for c in range(nc):
        for r in range(nr):
            i = r * nc + c
            plt.subplot(nr, nc, i + 1)
            plt.imshow(imgs[r][c])
            plt.axis("off")

    plt.tight_layout()
    path = "visual/{}.png".format(img_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)


def get_logger(date_str):
    log_fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = "visual/{}/train.log".format(date_str)
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
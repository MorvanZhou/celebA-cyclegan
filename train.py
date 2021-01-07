import os
import time
from dataset import load_celebA_tfrecord
from cyclegan import CycleGAN
import utils
import argparse
import tensorflow as tf
import datetime
import numpy as np

tf.random.set_seed(1)
np.random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", dest="batch_size", default=5, type=int)
parser.add_argument("-e", "--epoch", dest="epoch", default=101, type=int)
parser.add_argument("--soft_gpu", dest="soft_gpu", action="store_true", default=False)
parser.add_argument("--identity", dest="identity", action="store_true", default=False)
parser.add_argument("--cycle_lambda", dest="cycle_lambda", default=5, type=float)
parser.add_argument("--gp_lambda", dest="gp_lambda", default=10, type=float)
parser.add_argument("-lr", "--learning_rate", dest="lr", default=0.0002, type=float)
parser.add_argument("-b1", "--beta1", dest="beta1", default=0., type=float)
parser.add_argument("-b2", "--beta2", dest="beta2", default=0.99, type=float)
parser.add_argument("--lsgan", dest="lsgan", action="store_true", default=False)
parser.add_argument("--data_dir", dest="data_dir", default="./data")

args = parser.parse_args(
    # """--data_dir data -b 3 --epoch 101 --cycle_lambda 10 --gp_lambda 10 -lr 0.0002 -b1 0. -b2 0.99""".split(" ")
)

date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def train(gan, d):
    _dir = "visual/{}/model".format(date_str)
    checkpoint_path = _dir + "/cp-{epoch:04d}-{step:08d}.ckpt"
    os.makedirs(_dir, exist_ok=True)
    t0 = time.time()
    test_men = np.concatenate([next(iter(d.ds_men)) for _ in range(max(1, 15 // args.batch_size))], axis=0)[:10]
    test_women = np.concatenate([next(iter(d.ds_women)) for _ in range(max(1, 15 // args.batch_size))], axis=0)[:10]
    for ep in range(args.epoch):
        for t, img_men in enumerate(d.ds_men):
            img_women = next(iter(d.ds_women))
            g_loss, d_loss, cyc_loss = gan.step(img_men, img_women)
            if t % 500 == 0:
                utils.save_gan(gan, "%s/ep%03dt%d" % (date_str, ep, t), test_women, test_men)
                t1 = time.time()
                logger.info(
                    "ep={:03d} t={:04d} | time={:05.1f} | g_loss={:.2f} | d_loss={:.2f} | cyc_loss={:.2f}".format(
                        ep, t, t1 - t0, g_loss.numpy(), d_loss.numpy(), cyc_loss.numpy()))
                t0 = t1
            if t % 2000 == 0:
                gan.save_weights(checkpoint_path.format(epoch=ep, step=t))
    gan.save_weights(checkpoint_path.format(epoch=args.epoch, step=0))


def init_logger(date_str, m):
    logger = utils.get_logger(date_str)
    logger.info(str(args))
    logger.info("model parameters: g12={}, g21={}, d1={}, d2={}".format(
        m.g12.count_params(), m.g21.count_params(), m.d1.count_params(), m.d2.count_params()))

    try:
        tf.keras.utils.plot_model(m.g12, show_shapes=True, expand_nested=True, dpi=150,
                                  to_file="visual/{}/net_g.png".format(date_str))
        tf.keras.utils.plot_model(m.d1, show_shapes=True, expand_nested=True, dpi=150,
                                  to_file="visual/{}/net_d.png".format(date_str))
    except Exception as e:
        print(e)
    return logger


if __name__ == "__main__":
    utils.set_soft_gpu(args.soft_gpu)

    summary_writer = tf.summary.create_file_writer('visual/{}'.format(date_str))
    d = load_celebA_tfrecord(args.batch_size, args.data_dir)
    m = CycleGAN(img_shape=(128, 128, 3), cycle_lambda=args.cycle_lambda, summary_writer=summary_writer,
                 lr=args.lr, beta1=args.beta1, beta2=args.beta2, use_identity=args.identity, ls_loss=args.lsgan)
    logger = init_logger(date_str, m)
    train(m, d)



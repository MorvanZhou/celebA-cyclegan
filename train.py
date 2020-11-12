import os
import time
from dataset import load_celebA_tfrecord
from acgan import ACGAN
from acgangp import ACGANgp
import utils
import argparse
import tensorflow as tf
import datetime


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", dest="model", default="acgan")
parser.add_argument("-z", "--latent_dim", dest="latent_dim", default=200, type=int)
parser.add_argument("-l", "--label_dim", dest="label_dim", default=40, type=int)
parser.add_argument("-b", "--batch_size", dest="batch_size", default=25, type=int)
parser.add_argument("-e", "--epoch", dest="epoch", default=101, type=int)
parser.add_argument("--soft_gpu", dest="soft_gpu", action="store_true", default=True)
parser.add_argument("--lambda", dest="lambda_", default=10, type=float)
parser.add_argument("--d_loop", dest="d_loop", default=1, type=int)
parser.add_argument("-lr", "--learning_rate", dest="lr", default=0.0002, type=float)
parser.add_argument("-b1", "--beta1", dest="beta1", default=0., type=float)
parser.add_argument("-b2", "--beta2", dest="beta2", default=0.9, type=float)
parser.add_argument("--net", dest="net", default="dcnet", type=str, help="dcnet or resnet")

args = parser.parse_args()

date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def train(gan, d):
    _dir = "visual/{}/{}/model".format(model_name, date_str)
    checkpoint_path = _dir + "/cp-{epoch:04d}.ckpt"
    os.makedirs(_dir, exist_ok=True)
    t0 = time.time()
    for ep in range(args.epoch):
        for t, (real_img, real_img_label) in enumerate(d.ds):
            d_loss, g_loss = gan.step(real_img, real_img_label)

            if t % 1000 == 0:
                t1 = time.time()
                utils.save_gan(gan, "%s/ep%03dt%06d" % (date_str, ep, t))
                logger.info("ep={:03d} t={:04d}| time={:05.1f} | d_loss={:.2f} | g_loss={:.2f}".format(
                    ep, t, t1 - t0, d_loss.numpy(), g_loss.numpy()))
                t0 = t1
        if (ep + 1) % 5 == 0:
            gan.save_weights(checkpoint_path.format(epoch=ep))
    gan.save_weights(checkpoint_path.format(epoch=args.epoch))


def traingp(gan, d):
    steps = 202599 // args.batch_size
    _dir = "visual/{}/{}/model".format(model_name, date_str)
    checkpoint_path = _dir + "/cp-{epoch:04d}.ckpt"
    os.makedirs(_dir, exist_ok=True)
    t0 = time.time()
    for ep in range(args.epoch):
        for t in range(steps):
            g_loss = gan.train_g(args.batch_size)
            for _ in range(args.d_loop):
                real_img, real_img_label = next(iter(d.ds))
                w_loss, gp_loss, class_loss = gan.train_d(real_img, real_img_label)

            if t % 1000 == 0:
                utils.save_gan(gan, "%s/ep%03dt%06d" % (date_str, ep, t))
                t1 = time.time()
                logger.info("ep={:03d} t={:04d} | time={:05.1f} | w_loss={:.2f} | gp={:.2f} | class={:.2f} | g_loss={:.2f}".format(
                    ep, t, t1-t0, w_loss.numpy(), gp_loss.numpy(), class_loss.numpy(), g_loss.numpy()))
                t0 = t1
        if (ep+1) % 5 == 0:
            gan.save_weights(checkpoint_path.format(epoch=ep))
    gan.save_weights(checkpoint_path.format(epoch=args.epoch))


def init_logger(model_name, date_str, m):
    logger = utils.get_logger(model_name, date_str)
    logger.info(str(args))
    logger.info("model parameters: g={}, d={}".format(m.g.count_params(), m.d.count_params()))

    try:
        tf.keras.utils.plot_model(m.g, show_shapes=True, expand_nested=True, dpi=150,
                                  to_file="visual/{}/{}/net_g.png".format(model_name, date_str))
        tf.keras.utils.plot_model(m.d, show_shapes=True, expand_nested=True, dpi=150,
                                  to_file="visual/{}/{}/net_d.png".format(model_name, date_str))
    except Exception:
        pass
    return logger


if __name__ == "__main__":
    utils.set_soft_gpu(args.soft_gpu)

    model_name = args.model
    summary_writer = tf.summary.create_file_writer('visual/{}/{}'.format(model_name, date_str))
    if model_name == "acgan":
        d = load_celebA_tfrecord(args.batch_size // 2)
        m = ACGAN(latent_dim=args.latent_dim, label_dim=args.label_dim, img_shape=(128, 128, 3), a=-1, b=1, c=1,
                  summary_writer=summary_writer, lr=args.lr, beta1=args.beta1, beta2=args.beta2, net=args.net)
        logger = init_logger(model_name, date_str, m)
        train(m, d)
    elif model_name == "acgangp":
        d = load_celebA_tfrecord(args.batch_size)
        m = ACGANgp(args.latent_dim, args.label_dim, (128, 128, 3), args.lambda_,
                    summary_writer=summary_writer, lr=args.lr, beta1=args.beta1, beta2=args.beta2, net=args.net)
        logger = init_logger(model_name, date_str, m)
        traingp(m, d)
    else:
        raise ValueError("model name error")

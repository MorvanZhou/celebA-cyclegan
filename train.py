import os
import time
from dataset import load_celebA_tfrecord
from cyclegan import CycleGAN
import utils
import argparse
import tensorflow as tf
import datetime


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", dest="batch_size", default=1, type=int)
parser.add_argument("-e", "--epoch", dest="epoch", default=101, type=int)
parser.add_argument("--soft_gpu", dest="soft_gpu", action="store_true", default=False)
parser.add_argument("--identity", dest="identity", action="store_true", default=False)
parser.add_argument("--lambda", dest="lambda_", default=10, type=float)
parser.add_argument("-lr", "--learning_rate", dest="lr", default=0.0002, type=float)
parser.add_argument("-b1", "--beta1", dest="beta1", default=0.5, type=float)
parser.add_argument("-b2", "--beta2", dest="beta2", default=0.999, type=float)
parser.add_argument("--lsgan", dest="lsgan", action="store_true", default=False)
parser.add_argument("--output_dir", dest="output_dir", default="./visual")
parser.add_argument("--data_dir", dest="data_dir", default="./data")

args = parser.parse_args()

date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def train(gan, d):
    _dir = "{}/{}/{}/model".format(args.output_dir, model_name, date_str)
    checkpoint_path = _dir + "/cp-{epoch:04d}.ckpt"
    os.makedirs(_dir, exist_ok=True)
    t0 = time.time()
    for ep in range(args.epoch):
        for t, img_men in enumerate(d.ds_men):
            img_women = next(iter(d.ds_women))
            g_loss, d_loss = gan.step(img_men, img_women)
            if t % 500 == 0:
                utils.save_gan(gan, "%s/ep%03dt%06d" % (date_str, ep, t), args.output_dir, img_women, img_men)
                t1 = time.time()
                logger.info("ep={:03d} t={:04d} | time={:05.1f} | g_loss={:.2f} | d_loss={:.2f}".format(
                    ep, t, t1-t0, g_loss.numpy(), d_loss.numpy()))
                t0 = t1
        if (ep+1) % 5 == 0:
            gan.save_weights(checkpoint_path.format(epoch=ep))
    gan.save_weights(checkpoint_path.format(epoch=args.epoch))


def init_logger(model_name, date_str, m):
    logger = utils.get_logger(model_name, date_str)
    logger.info(str(args))
    logger.info("model parameters: g={}, f={}, dg={}, df={}".format(
        m.g.count_params(), m.f.count_params(), m.dg.count_params(), m.df.count_params()))

    try:
        tf.keras.utils.plot_model(m.g, show_shapes=True, expand_nested=True, dpi=150,
                                  to_file="{}/{}/{}/net_g.png".format(args.output_dir, model_name, date_str))
        tf.keras.utils.plot_model(m.dg, show_shapes=True, expand_nested=True, dpi=150,
                                  to_file="{}/{}/{}/net_d.png".format(args.output_dir, model_name, date_str))
    except Exception as e:
        print(e)
    return logger


if __name__ == "__main__":
    utils.set_soft_gpu(args.soft_gpu)

    model_name = "cyclegan"
    summary_writer = tf.summary.create_file_writer('{}/{}/{}'.format(args.output_dir, model_name, date_str))
    d = load_celebA_tfrecord(args.batch_size)
    m = CycleGAN(img_shape=(128, 128, 3), lambda_=args.lambda_, summary_writer=summary_writer,
                 lr=args.lr, beta1=args.beta1, beta2=args.beta2, use_identity=args.identity, ls_loss=args.lsgan)
    logger = init_logger(model_name, date_str, m)
    train(m, d)



import os
import time
from dataset import load_celebA_tfrecord
from cyclegan import CycleGAN
import utils
import argparse
import tensorflow as tf
import datetime


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", dest="batch_size", default=10, type=int)
parser.add_argument("--soft_gpu", dest="soft_gpu", action="store_true", default=False)
parser.add_argument("--output_dir", dest="output_dir", default="./visual")
parser.add_argument("--data_dir", dest="data_dir", default="./data")
parser.add_argument("--ep", dest="ep", default="last")
parser.add_argument("--step", dest="step", default="last")
parser.add_argument("--model_path", )

args = parser.parse_args()


def train(gan, d):
    gan.load_weights(args.model_path)
    test_men = next(iter(d.ds_men))
    test_women = next(iter(d.ds_women))
    utils.save_gan(gan, "ep%03dt%06d" % (args.ep, args.step), args.output_dir, test_women, test_men)


if __name__ == "__main__":
    utils.set_soft_gpu(args.soft_gpu)

    d = load_celebA_tfrecord(args.batch_size)
    m = CycleGAN(img_shape=(128, 128, 3), lambda_=args.lambda_)

    train(m, d)



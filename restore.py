import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from cyclegan import CycleGAN


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("-t", "--transfer", help="f2m (female to male) or m2f (male to female)", default="m2f")
parser.add_argument("--image_dir", default="demo/male", type=str)

args = parser.parse_args(
    # """--model_path visual/2020-12-21_15-54-47/model/cp-0003-00006000.ckpt -t m2f --image_dir demo/male""".split(" ")
    """--model_path visual/2020-12-21_15-54-47/model/cp-0003-00006000.ckpt -t f2m --image_dir demo/female""".split(" ")
)


def generate(generator):
    for img_name in os.listdir(args.image_dir):
        if args.transfer == "m2f" and not img_name.startswith("male"):
            continue
        if args.transfer == "f2m" and not img_name.startswith("female"):
            continue
        if img_name.endswith("transfer.png"):
            continue
        img_path = os.path.join(args.image_dir, img_name)
        img = Image.open(img_path)

        w, h = img.size
        min_l = min(w, h)
        ws = int(w/2-min_l/2)
        hs = int(h / 2 - min_l / 2)
        img = img.crop((ws, hs, ws+min_l, hs+min_l)).resize((128, 128), Image.ANTIALIAS)
        img = np.array(img, dtype=np.float32) / 255 * 2 - 1
        if img.shape[-1] != 3:
            print("channel not 3 ", img_path)
            continue
        img_ = generator.predict(np.expand_dims(img, axis=0))  # man to woman


        img_ = np.squeeze(img_, axis=0)
        img_ = (img_ + 1) / 2
        img = (img + 1) / 2
        plt.clf()
        plt.figure(0, (4, 2))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_)
        plt.axis("off")
        plt.tight_layout()
        dir_ = os.path.dirname(img_path)
        img_name = os.path.basename(img_path).split(".")[0]
        new_img_name = img_name+"_transfer"
        path = os.path.join(dir_, new_img_name)
        os.makedirs(dir_, exist_ok=True)
        plt.savefig(path)


if __name__ == "__main__":
    gan = CycleGAN(img_shape=(128, 128, 3))
    gan.load_weights(args.model_path)
    if args.transfer == "m2f":
        generate(gan.g12)
    elif args.transfer == "f2m":
        generate(gan.g21)
    else:
        raise ValueError("{} transformation is not supported".format(args.transfer))



import matplotlib.pyplot as plt
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# data is downloaded from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    values = value if isinstance(value, (list, tuple)) else [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _img_array_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value.ravel()))


def _bytes_img_process(img_str):
    crop = [20, 0, 178, 178]
    imgs = tf.io.decode_and_crop_jpeg(img_str, crop)
    imgs = tf.image.resize(imgs, (128, 128))
    return imgs


class CelebA:
    def __init__(self, batch_size, data_dir="data",):
        self.label_path = os.path.join(data_dir, "list_attr_celeba.txt")
        self.img_dir = os.path.join(data_dir, "img_align_celeba")
        self.tfrecord_dir = os.path.join(data_dir, "tfrecord-celebA-cyclegan")
        self.batch_size = batch_size

        self.ds_men = None
        self.ds_women = None

    def _image_example(self, img):
        feature = {
            "img": _bytes_feature(img),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def _parse_img(self, example_proto):
        feature = tf.io.parse_single_example(example_proto, features={
            "img": tf.io.FixedLenFeature([], tf.string)
        })
        imgs = _bytes_img_process(feature["img"])
        return tf.cast(imgs, tf.float32) / 255 * 2 - 1

    def load_tf_recoder(self):
        ds = []
        name = ["men", "women"]
        for i in range(2):
            dir_ = os.path.join(self.tfrecord_dir, name[i])
            paths = [os.path.join(dir_, p) for p in os.listdir(dir_)]
            raw_img_ds = tf.data.TFRecordDataset(paths, num_parallel_reads=min(4, len(paths)))
            ds.append(raw_img_ds.shuffle(1024).map(
                    self._parse_img, num_parallel_calls=tf.data.experimental.AUTOTUNE,
                ).batch(
                    self.batch_size, drop_remainder=True
                ).prefetch(
                    tf.data.experimental.AUTOTUNE
                ))
        self.ds_men = ds[0]
        self.ds_women = ds[1]

    def to_tf_recoder(self):
        with open(self.label_path) as f:
            lines = f.readlines()
            all_labels = lines[1].strip().split(" ")
            male_label_id = all_labels.index("Male")
            lines = lines[2:]
            n = 202599//3
            chunks = [lines[i:i + n] for i in range(0, len(lines), n)]
            for i, chunk in enumerate(chunks):
                men_path = os.path.join(self.tfrecord_dir, "men/{}.tfrecord".format(i))
                women_path = os.path.join(self.tfrecord_dir, "women/{}.tfrecord".format(i))
                os.makedirs(os.path.dirname(men_path), exist_ok=True)
                os.makedirs(os.path.dirname(women_path), exist_ok=True)
                with tf.io.TFRecordWriter(men_path) as men_writer:
                    with tf.io.TFRecordWriter(women_path) as women_writer:
                        for line in chunk:
                            img_name, img_labels = line.split(" ", 1)
                            try:
                                img = open(os.path.join(self.img_dir, img_name), "rb").read()
                            except Exception as e:
                                break
                            label_str = img_labels.replace("  ", " ").split(" ")
                            is_woman = label_str[male_label_id] == "-1"
                            tf_example = self._image_example(img).SerializeToString()
                            if is_woman:
                                women_writer.write(tf_example)
                            else:
                                men_writer.write(tf_example)


def show_sample(data_dir):
    d = load_celebA_tfrecord(5, data_dir)
    images = tf.concat([next(iter(d.ds_women)), next(iter(d.ds_men))], axis=0)
    images = (images.numpy() + 1) / 2
    for i in range(2):
        for j in range(5):
            n = i*5+j
            plt.subplot(2, 5, n+1)
            plt.imshow(images[n])
            plt.xticks(())
            plt.yticks(())
    plt.show()


def parse_celebA_tfreord(data_dir):
    d = CelebA(1, data_dir)
    d.to_tf_recoder()


def load_celebA_tfrecord(batch_size, data_dir):
    d = CelebA(batch_size, data_dir)
    d.load_tf_recoder()
    return d


if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default="data", type=str)

    args = parser.parse_args()

    DATA_DIR = args.data_dir

    t0 = time.time()
    parse_celebA_tfreord(DATA_DIR)
    # ds = load_celebA_tfrecord(20, DATA_DIR)
    # t1 = time.time()
    # print("load time", t1-t0)
    # count = 0
    # while True:
    #     for img, label in ds:
    #         # if _ % 200 == 0:
    #         count+=1
    #         if count % 500==0: print(img.shape, label.shape)
    #         if count == 10000:
    #             break
    #     if count == 10000:
    #         break
    #
    # print("runtime", time.time()-t1)
    show_sample(DATA_DIR)

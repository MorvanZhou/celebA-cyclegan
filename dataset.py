import matplotlib.pyplot as plt
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# data is downloaded from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

LABEL_PATH = "data/list_attr_celeba.txt"
IMAGE_DIR = "data/img_align_celeba"
IMG_SHAPE = (128, 128, 3)

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
    crop = [45, 25, 128, 128]
    imgs = tf.io.decode_and_crop_jpeg(img_str, crop)
    return imgs


def _int_img_process(img_int):
    imgs = tf.reshape(img_int, IMG_SHAPE)
    return imgs


class CelebA:
    def __init__(self, batch_size, image_size=(128, 128, 3),
                 label_path="data/list_attr_celeba.txt", img_dir="data/img_align_celeba"):
        self.label_path = label_path
        self.img_dir = img_dir
        self.batch_size = batch_size

        with open(label_path) as f:
            lines = f.readlines()
            all_labels = lines[1].strip().split(" ")
            self.male_label_id = all_labels.index("Male")
        self.image_size = image_size
        self.crop = [45, 45+image_size[0], 25, 25+image_size[1]]
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
        raw_img_ds = []
        for m in ["men", "women"]:
            dir_ = os.path.join(os.path.dirname(self.img_dir), "tfrecord", m)
            paths = [os.path.join(dir_, p) for p in os.listdir(dir_)]
            raw_img_ds.append(tf.data.TFRecordDataset(paths, num_parallel_reads=min(4, len(paths))))
        self.ds_men = raw_img_ds[0].shuffle(1024).map(
            self._parse_img, num_parallel_calls=tf.data.experimental.AUTOTUNE,
        ).batch(
            self.batch_size, drop_remainder=True
        ).prefetch(
            tf.data.experimental.AUTOTUNE
        )
        self.ds_women = raw_img_ds[1].shuffle(1024).map(
            self._parse_img, num_parallel_calls=tf.data.experimental.AUTOTUNE,
        ).batch(
            self.batch_size, drop_remainder=True
        ).prefetch(
            tf.data.experimental.AUTOTUNE
        )

    def to_tf_recoder(self):
        with open(self.label_path) as f:
            lines = f.readlines()[2:]
            n = 202599//3
            chunks = [lines[i:i + n] for i in range(0, len(lines), n)]
            for i, chunk in enumerate(chunks):
                men_path = os.path.dirname(self.img_dir) + "/tfrecord/men/{}.tfrecord".format(i)
                women_path = os.path.dirname(self.img_dir) + "/tfrecord/women/{}.tfrecord".format(i)
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
                            is_woman = label_str[self.male_label_id] == "-1"
                            tf_example = self._image_example(img).SerializeToString()
                            if is_woman:
                                women_writer.write(tf_example)
                            else:
                                men_writer.write(tf_example)


def show_sample():
    d = load_celebA_tfrecord(5)
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


def parse_celebA_tfreord():
    d = CelebA(1, IMG_SHAPE, LABEL_PATH, IMAGE_DIR)
    d.to_tf_recoder()


def load_celebA_tfrecord(batch_size):
    d = CelebA(batch_size, IMG_SHAPE, LABEL_PATH, IMAGE_DIR)
    d.load_tf_recoder()
    return d


if __name__ == "__main__":
    import time

    t0 = time.time()
    # parse_celebA_tfreord()
    # ds = load_celebA_tfrecord(20)
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
    show_sample()

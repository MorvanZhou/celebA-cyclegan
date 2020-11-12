import tensorflow as tf
from tensorflow import keras
import numpy as np
from cnn import unet, resnet_d


class CycleGAN(keras.Model):
    def __init__(self, img_shape,
                 lambda_=10, summary_writer=None, lr=0.0001, beta1=0.5, beta2=0.99):
        super().__init__()
        self.img_shape = img_shape
        self.lambda_ = lambda_
        self.g = self._get_generator("g")
        self.f = self._get_generator("f")
        self.dg, self.patch_shape = self._get_discriminator("dg")
        self.df, _ = self._get_discriminator("df")

        self.opt = keras.optimizers.Adam(lr, beta_1=beta1, beta_2=beta2)
        self.loss_bool = keras.losses.BinaryCrossentropy(from_logits=True)
        self.loss_img = keras.losses.MeanAbsoluteError()

        self.summary_writer = summary_writer
        self._train_step = 0

    def _get_discriminator(self, name):
        model = resnet_d(self.img_shape, use_bn=True, name=name)  # [n, 4, 4, 1]
        model.summary()
        return model, tf.shape(model.layers[-1])[1:]

    def _get_generator(self, name):
        model = unet(self.img_shape, name)
        model.summary()
        return model

    def train_d(self, fimg, gimg, label):
        with tf.GradientTape() as tape:
            loss = (self.loss_bool(label, self.dg(gimg)) + self.loss_bool(label, self.df(fimg))) / 2
        vars = self.df.trainable_variables + self.dg.trainable_variables
        grads = tape.gradient(loss, vars)
        self.opt.apply_gradients(zip(grads, vars))
        return loss

    def cycle(self, img1, img2):
        gimg1, fimg2 = self.g(img1), self.f(img2)
        loss1 = self.loss_img(img1, self.f(gimg1))
        loss2 = self.loss_img(img2, self.g(fimg2))
        loss = self.lambda_ * (loss1 + loss2) / 2
        return loss, gimg1, fimg2

    def identity(self, img1, img2):
        loss1 = self.loss_img(img2, self.g(img2))
        loss2 = self.loss_img(img1, self.f(img1))
        return self.lambda_ * (loss1 + loss2) / 2

    def train_g(self, img1, img2):
        d_label = tf.ones((len(img1), *self.patch_shape), tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            cycle_loss, gimg1, fimg2 = self.cycle(img1, img2)
            identity_loss = self.identity(img1, img2)
            loss_g = self.loss_bool(d_label, self.dg(gimg1)) + cycle_loss + identity_loss
            loss_f = self.loss_bool(d_label, self.df(fimg2)) + cycle_loss + identity_loss
        grads_g = tape.gradient(loss_g, self.g.trainable_variables)
        grads_f = tape.gradient(loss_f, self.f.trainable_variables)
        self.opt.apply_gradients(zip(grads_g, self.g.trainable_variables))
        self.opt.apply_gradients(zip(grads_f, self.f.trainable_variables))
        del tape

        half = len(img1) // 2
        return (loss_g + loss_f) / 2, gimg1[:half], fimg2[:half]

    def step(self, img1, img2):
        g_loss, gimg1, fimg2 = self.train_g(img1, img2)

        half = len(fimg2)
        d_label = tf.concat(
            (tf.ones((half, *self.patch_shape), tf.float32),
             tf.zeros((half, *self.patch_shape), tf.float32)), axis=0)
        real_fake_fimg = tf.concat((img1[:half], fimg2), axis=0)
        real_fake_gimg = tf.concat((img2[:half], gimg1), axis=0)
        d_loss = self.train_d(real_fake_fimg, real_fake_gimg, d_label)
        return g_loss, d_loss

    def train_d(self, img, img_label):
        with tf.GradientTape() as tape:
            g_img = self.call(img_label, training=False)
            gp = self.gp(img, g_img)
            all_img = tf.concat((img, g_img), axis=0)
            pred, pred_class = self.d.call(all_img, training=True)
            loss_class = self.loss_class(tf.concat((img_label, img_label), axis=0), pred_class)
            pred_real, pred_fake = tf.split(pred, num_or_size_splits=2, axis=0)
            w_distance = tf.reduce_mean(pred_real) - tf.reduce_mean(pred_fake)  # maximize W distance
            gp_loss = self.lambda_ * gp
            loss = gp_loss + loss_class - w_distance
        grads = tape.gradient(loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))

        if self._train_step % 300 == 0 and self.summary_writer is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar("d/w_distance", w_distance, step=self._train_step)
                tf.summary.scalar("d/gp", gp_loss, step=self._train_step)
                tf.summary.scalar("d/sigmoid", loss_class, step=self._train_step)
                tf.summary.histogram("d/pred_real", pred_real, step=self._train_step)
                tf.summary.histogram("d/pred_fake", pred_fake, step=self._train_step)
                tf.summary.histogram("d/last_grad", grads[-1], step=self._train_step)
                tf.summary.histogram("d/first_grad", grads[0], step=self._train_step)
        return w_distance, gp_loss, loss_class

    def train_g(self, batch_size):
        random_img_label = tf.convert_to_tensor(
            np.random.choice([0, 1], (batch_size, self.label_dim), replace=True), dtype=tf.int32)
        with tf.GradientTape() as tape:
            g_img = self.call(random_img_label, training=True)
            pred_fake, pred_class = self.d.call(g_img, training=False)
            loss_class = self.loss_class(random_img_label, pred_class)
            w_distance = tf.reduce_mean(-pred_fake)  # minimize W distance
            loss = w_distance + loss_class
        grads = tape.gradient(loss, self.g.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.g.trainable_variables))

        if self._train_step % 300 == 0 and self.summary_writer is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar("g/w_distance", w_distance, step=self._train_step)
                tf.summary.scalar("g/sigmoid", loss_class, step=self._train_step)
                tf.summary.histogram("g/pred_fake", pred_fake, step=self._train_step)
                tf.summary.histogram("g/first_grad", grads[0], step=self._train_step)
                tf.summary.histogram("g/last_grad", grads[-1], step=self._train_step)
                if self._train_step % 1000 == 0:
                    tf.summary.image("g/img", (g_img + 1) / 2, max_outputs=5, step=self._train_step)
        self._train_step += 1
        return loss

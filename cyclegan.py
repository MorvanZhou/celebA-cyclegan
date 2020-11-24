import tensorflow as tf
from tensorflow import keras
from cnn import unet, dc_d
import random


class CycleGAN(keras.Model):
    def __init__(self, img_shape,
                 lambda_=10, summary_writer=None, lr=0.0001, beta1=0.5, beta2=0.99, use_identity=False, ls_loss=True):
        super().__init__()
        self.img_shape = img_shape
        self.lambda_ = lambda_
        self.use_identity = use_identity
        self.ls_loss = ls_loss
        self.g = self._get_generator("g")       # man to woman
        self.f = self._get_generator("f")       # woman to man
        self.dg, self.patch_shape = self._get_discriminator("dg")
        self.df, _ = self._get_discriminator("df")
        self.g_buffer = []
        self.f_buffer = []
        self.buffer_len = 10

        self.opt = keras.optimizers.Adam(lr, beta_1=beta1, beta_2=beta2)
        self.loss_img = keras.losses.MeanAbsoluteError()
        self.d_loss_fun = keras.losses.MeanSquaredError() if self.ls_loss \
            else keras.losses.BinaryCrossentropy(from_logits=True)

        self.summary_writer = summary_writer
        self._train_step = 0

    def _get_discriminator(self, name):
        model = dc_d(self.img_shape, name=name)  # [n, 8, 8, 1]
        model.summary()
        return model, model.layers[-1].output_shape[1:]

    def _get_generator(self, name):
        model = unet(self.img_shape, name)
        model.summary()
        return model

    def train_d(self, fimg, gimg, label):
        with tf.GradientTape() as tape:
            loss = (self.d_loss_fun(label, self.dg(gimg)) + self.d_loss_fun(label, self.df(fimg))) / 4
        vars = self.df.trainable_variables + self.dg.trainable_variables
        grads = tape.gradient(loss, vars)
        self.opt.apply_gradients(zip(grads, vars))

        if self._train_step % 300 == 0 and self.summary_writer is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar("d/loss", loss, step=self._train_step)
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
        return 0.5 * self.lambda_ * (loss1 + loss2) / 2

    def train_g(self, img1, img2):
        with tf.GradientTape(persistent=True) as tape:
            cycle_loss, gimg1, fimg2 = self.cycle(img1, img2)
            if self.use_identity:
                cycle_loss += self.identity(img1, img2)
            loss_g = self.d_loss_fun(1, self.dg(gimg1)) + cycle_loss
            loss_f = self.d_loss_fun(1, self.df(fimg2)) + cycle_loss
        grads_g = tape.gradient(loss_g, self.g.trainable_variables)
        grads_f = tape.gradient(loss_f, self.f.trainable_variables)
        self.opt.apply_gradients(zip(grads_g, self.g.trainable_variables))
        self.opt.apply_gradients(zip(grads_f, self.f.trainable_variables))
        del tape

        if self._train_step % 300 == 0 and self.summary_writer is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar("g/cycle_loss", cycle_loss, step=self._train_step)
                tf.summary.histogram("g/loss_g", loss_g, step=self._train_step)
                tf.summary.histogram("g/loss_f", loss_f, step=self._train_step)
                if self._train_step % 1000 == 0:
                    tf.summary.image("gimg", (gimg1 + 1) / 2, max_outputs=5, step=self._train_step)
                    tf.summary.image("fimg", (fimg2 + 1) / 2, max_outputs=5, step=self._train_step)

        half = len(img1) // 2
        return (loss_g + loss_f) / 2, gimg1[:half], fimg2[:half]

    def step(self, img1, img2):
        g_loss, gimg1, fimg2 = self.train_g(img1, img2)

        half = len(fimg2)
        if self.ls_loss:
            d_label = tf.concat(
                (tf.ones((half, *self.patch_shape), tf.float32),    # real
                 -tf.ones((half, *self.patch_shape), tf.float32)), axis=0   # fake
            )
        else:
            d_label = tf.concat(
                (tf.ones((half, *self.patch_shape), tf.float32),
                 tf.zeros((half, *self.patch_shape), tf.float32)), axis=0)

        # reduce model oscillation
        self.g_buffer.append(gimg1)
        self.f_buffer.append(fimg2)
        if len(self.g_buffer) > self.buffer_len:
            self.g_buffer.pop(0)
        if len(self.f_buffer) > self.buffer_len:
            self.f_buffer.pop(0)
        idx = random.randint(0, len(self.f_buffer)-1)
        gimg1, fimg2 = self.g_buffer[idx], self.f_buffer[idx]

        real_fake_fimg = tf.concat((img1[:half], fimg2), axis=0)
        real_fake_gimg = tf.concat((img2[:half], gimg1), axis=0)
        d_loss = self.train_d(real_fake_fimg, real_fake_gimg, d_label)
        self._train_step += 1
        return g_loss, d_loss
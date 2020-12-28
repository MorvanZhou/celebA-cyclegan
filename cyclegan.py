import tensorflow as tf
from tensorflow import keras
from cnn import unet, dc_d

fake1_buffer = []
fake2_buffer = []


class CycleGAN(keras.Model):
    def __init__(self, img_shape,
                 cycle_lambda=10, summary_writer=None, lr=0.0001, beta1=0.5, beta2=0.99, gp_lambda=10, use_identity=False, ls_loss=True):
        super().__init__()
        self.img_shape = img_shape
        self.cycle_lambda = cycle_lambda
        self.gp_lambda = gp_lambda
        self.use_identity = use_identity
        self.ls_loss = ls_loss
        self.g12 = self._get_generator("g12")       # man to woman
        self.g21 = self._get_generator("g21")       # woman to man
        self.d1, self.patch_shape = self._get_discriminator("d1")
        self.d2, _ = self._get_discriminator("d2")

        self.buffer_len = 20

        self.opt = keras.optimizers.Adam(lr, beta_1=beta1, beta_2=beta2)
        self.loss_img = keras.losses.MeanAbsoluteError()
        self.d_loss_fun = keras.losses.MeanSquaredError() if self.ls_loss \
            else keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)

        self.summary_writer = summary_writer
        self._train_step = 0

    def _get_discriminator(self, name):
        model = dc_d(self.img_shape, name=name)  # [n, 8, 8, 1]
        if not self.ls_loss:
            model.add(keras.layers.GlobalAveragePooling2D())
            model.add(keras.layers.Dense(1))
        else:
            model.add(keras.layers.Conv2D(1, 2, 1, "valid"))
        model.summary()
        return model, model.layers[-1].output_shape[1:]

    def _get_generator(self, name):
        model = unet(self.img_shape, name)
        model.summary()
        return model

    # gradient penalty
    def gp(self, real_img, fake_img, d):
        e = tf.random.uniform((len(real_img), 1, 1, 1), 0, 1)
        noise_img = e * real_img + (1. - e) * fake_img  # extend distribution space
        with tf.GradientTape() as tape:
            tape.watch(noise_img)
            o = d(noise_img)
        g = tape.gradient(o, noise_img)  # image gradients
        g_norm2 = tf.sqrt(tf.reduce_sum(tf.square(g), axis=[1, 2, 3]))  # norm2 penalty
        gp = tf.square(g_norm2 - 1.)
        return tf.reduce_mean(gp)

    @staticmethod
    def w_distance(real, fake):
        # the distance of two data distributions
        return tf.reduce_mean(real) - tf.reduce_mean(fake)

    def train_d(self, real1, real2):
        d_w_distance = 0
        d_gp = 0
        for real, real_, g, g_, d in zip(
                [real1, real2], [real2, real1],
                [self.g12, self.g21], [self.g21, self.g12], [self.d2, self.d1]
        ):
            with tf.GradientTape() as tape:
                fake_style = g(real)
                pred_fake, pred_real = d(fake_style), d(real_)
                w_distance = -self.w_distance(pred_real, pred_fake)  # maximize W distance
                gp = self.gp(real_, fake_style, d)
                loss = w_distance + self.gp_lambda * gp
            grads = tape.gradient(loss, d.trainable_variables)
            self.opt.apply_gradients(zip(grads, d.trainable_variables))
            d_gp += gp
            d_w_distance += w_distance

        if self._train_step % 300 == 0 and self.summary_writer is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar("d/w_distance", d_w_distance, step=self._train_step)
                tf.summary.scalar("d/gp", d_gp, step=self._train_step)
        return loss

    def cycle(self, real, g, g_):
        fake_style = g(real)
        fake_real = g_(fake_style)
        loss = self.loss_img(real, fake_real)
        return loss, fake_style, fake_real

    def identity(self, real, g):
        loss = self.loss_img(real, g(real))
        return loss

    def train_g(self, real1, real2):
        cyc_losses, d_losses = 0, 0
        fake_styles = []
        for real, g, g_, d in zip(
                [real1, real2],
                [self.g12, self.g21], [self.g21, self.g12], [self.d2, self.d1]
        ):
            with tf.GradientTape() as tape:
                # real -> fake_style -> fake_real         autoencoder
                # cyc_loss = abs(real - fake_real)
                cyc_loss, fake_style, fake_real = self.cycle(real, g, g_)
                # is fake_real real?
                pred = d(fake_style)
                d_loss = self.d_loss_fun(tf.ones_like(pred), pred)  # make style transfer more real
                loss = d_loss + self.cycle_lambda * cyc_loss
                if self.use_identity:
                    loss += self.cycle_lambda / 5 * self.identity(real, g)
            var = self.g12.trainable_variables + self.g21.trainable_variables
            grads = tape.gradient(loss, var)
            self.opt.apply_gradients(zip(grads, var))
            cyc_losses += cyc_loss
            d_losses += d_loss
            fake_styles.append(fake_style)

        if self._train_step % 300 == 0 and self.summary_writer is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar("g/cycle_loss", cyc_losses, step=self._train_step)
                tf.summary.scalar("g/d_loss", d_losses, step=self._train_step)
        return d_losses, cyc_losses

    def step(self, real1, real2):
        g_loss, cyc_loss = self.train_g(real1, real2)
        d_loss = self.train_d(real1, real2)
        self._train_step += 1
        return g_loss, d_loss, cyc_loss
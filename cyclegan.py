import tensorflow as tf
from tensorflow import keras
from cnn import unet, dc_d

fake1_buffer = []
fake2_buffer = []


class CycleGAN(keras.Model):
    def __init__(self, img_shape,
                 lambda_=10, summary_writer=None, lr=0.0001, beta1=0.5, beta2=0.99, use_identity=False, ls_loss=True):
        super().__init__()
        self.img_shape = img_shape
        self.lambda_ = lambda_
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
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(1))
        model.summary()
        return model, model.layers[-1].output_shape[1:]

    def _get_generator(self, name):
        model = unet(self.img_shape, name)
        model.summary()
        return model

    def train_d(self, real_fake1, real_fake2, label):
        with tf.GradientTape() as tape:
            loss = self.d_loss_fun(label, self.d1(real_fake1)) + self.d_loss_fun(label, self.d2(real_fake2))
        var = self.d1.trainable_variables + self.d2.trainable_variables
        grads = tape.gradient(loss, var)
        self.opt.apply_gradients(zip(grads, var))

        if self._train_step % 300 == 0 and self.summary_writer is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar("d/loss", loss, step=self._train_step)
        return loss

    def cycle(self, real, g, g_):
        fake_style = g(real)
        fake_real = g_(fake_style)
        loss = self.loss_img(real, fake_real)
        return loss, fake_style, fake_real

    def identity(self, real, g):
        loss = self.loss_img(real, g(real))
        # loss12 = self.loss_img(real1, self.g12(real1))
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
                loss = d_loss + self.lambda_ * cyc_loss
                if self.use_identity:
                    loss += self.lambda_ / 5 * self.identity(real, g)
            var = self.g12.trainable_variables + self.g21.trainable_variables
            grads = tape.gradient(loss, var)
            self.opt.apply_gradients(zip(grads, var))
            cyc_losses += cyc_loss
            d_losses += d_loss
            fake_styles.append(fake_style)

        if self._train_step % 300 == 0 and self.summary_writer is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar("g/cycle_loss", cyc_losses, step=self._train_step)
                tf.summary.histogram("g/d_loss", d_losses, step=self._train_step)

        half = len(real1) // 2
        fake2 = fake_styles[0]
        fake1 = fake_styles[1]
        return d_losses, cyc_losses, fake2[:half], fake1[:half]

    def step(self, real1, real2):
        g_loss, cyc_loss, half_fake2, half_fake1 = self.train_g(real1, real2)

        half = len(half_fake2)
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
        # fake1_buffer.append(half_fake1.numpy())
        # fake2_buffer.append(half_fake2.numpy())
        # idx = random.randint(0, len(fake1_buffer)-1)
        # half_fake1, half_fake2 = tf.convert_to_tensor(fake1_buffer[idx]), tf.convert_to_tensor(fake2_buffer[idx])
        # if len(fake1_buffer) > self.buffer_len:
        #     fake1_buffer.pop(0)
        # if len(fake2_buffer) > self.buffer_len:
        #     fake2_buffer.pop(0)

        real_fake1 = tf.concat((real1[:half], half_fake1), axis=0)
        real_fake2 = tf.concat((real2[:half], half_fake2), axis=0)
        d_loss = self.train_d(real_fake1, real_fake2, d_label)
        self._train_step += 1
        return g_loss, d_loss, cyc_loss
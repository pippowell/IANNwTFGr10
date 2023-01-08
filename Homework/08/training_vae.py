import tensorflow as tf
import dataset
import model
import matplotlib.pyplot as plt
import datetime as datetime
import numpy as np

# prep dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

def preprocessing_vae(data):
    data = tf.cast(data, tf.float32)
    data = (data/128.) - 1

    # create noise to be added to image
    noise_factor = 0.5    
    noise = noise_factor*tf.random.normal(shape=data.shape, mean=0.5, stddev=0.5, dtype=tf.dtypes.float32)
    data = data + noise
    
    data = tf.clip_by_value(data, clip_value_min=-1, clip_value_max=1)
    data = tf.expand_dims(data, axis=-1) # from (28,28) to (28,28,1)

    return data

x_train = preprocessing_vae(x_train)
x_test = preprocessing_vae(x_test)

class VAE(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = model.encoder(embedding=10, vae=True)
        self.decoder = model.decoder()
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# class CustomVariationalLayer(Layer):
#     def __init__(self, **kwargs):
#         self.is_placeholder = True
#         super(CustomVariationalLayer, self).__init__(**kwargs)

#     def vae_loss(self, x, x_decoded_mean):
#         xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
#         kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#         return K.mean(xent_loss + kl_loss)

#     def call(self, inputs):
#         x = inputs[0]
#         x_decoded_mean = inputs[1]
#         loss = self.vae_loss(x, x_decoded_mean)
#         self.add_loss(loss, inputs=inputs)
#         # We won't actually use the output.
#         return x

# training
vae1 = VAE()
vae2 = model.autoencoder(vae=True)


def loss_func(encoder_mu, encoder_log_variance):
    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss_factor = 1000
        reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(y_true-y_predict), axis=[1, 2, 3])
        return reconstruction_loss_factor * reconstruction_loss

    def vae_kl_loss(encoder_mu, encoder_log_variance):
        kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(encoder_log_variance), axis=1)
        return kl_loss

    # def vae_kl_loss_metric(y_true, y_predict):
    #     kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(encoder_log_variance), axis=1)
    #     return kl_loss

    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)

        loss = reconstruction_loss + kl_loss
        return loss

    return vae_loss

epochs = 2
lr = 1e-3

opti = tf.keras.optimizers.Adam(learning_rate=lr)
loss_vae1 = tf.keras.losses.MeanSquaredError() # tf.keras.losses.BinaryCrossentropy()
loss_vae2 = loss_func(model.zmean, model.zlogvar)


# mnist_digits = np.concatenate([x_train, x_test], axis=0)

# not working
# vae1.compile(loss=loss_vae1, optimizer=opti)

# working, but high losses
vae2.compile(loss=loss_vae2, optimizer=opti)
history2 = vae2.fit(x_train, x_train, 
                validation_data=(x_test, x_test),
                epochs=epochs,
                batch_size=64,
                shuffle=True)

# plotting for vae2
plt.plot(history2.history["loss"])
plt.plot(history2.history["val_loss"])
plt.legend(labels=["train_loss", "val_loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(f"Homework/08/plot/VAE:e={epochs},lr={lr}.png")
plt.show()

# reference taken here: https://blog.paperspace.com/how-to-build-variational-autoencoder-keras/ 


# def mse_loss(y_true, y_pred):
    #         r_loss = K.mean(K.square(y_true - y_pred), axis = [1,2,3])
    #         return 1000 * r_loss

    # def kl_loss(mean, log_var):
    #     kl_loss =  -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis = 1)
    #     return kl_loss

    # def vae_loss(y_true, y_pred, mean, var):
    #     r_loss = mse_loss(y_true, y_pred)
    #     kl_loss = kl_loss(mean, log_var)
    #     return  r_loss + kl_loss
    
    # @tf.function
    # def train_step(images):
    
    #     with tf.GradientTape() as encoder, tf.GradientTape() as decoder:
    #         mean, log_var = encoder(images, training=True)
    #         latent = sampling([mean, log_var])
    #         generated_images = dec(latent, training=True)
    #         loss = vae_loss(images, generated_images, mean, log_var)
    
    #         gradients_of_enc = encoder.gradient(loss, enc.trainable_variables)
    #         gradients_of_dec = decoder.gradient(loss, dec.trainable_variables)

    #         opti.apply_gradients(zip(gradients_of_enc, enc.trainable_variables))
    #         opti.apply_gradients(zip(gradients_of_dec, dec.trainable_variables))
        
    #     return loss
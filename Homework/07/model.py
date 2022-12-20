
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np


class BasicConv(tf.keras.Model):
    def __init__(self, batch, sequence_length, image):
        super(BasicConv, self).__init__()

        # input 32x32x3 with 3 as the color channels
        self.convlayer1 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu') # after this: 32x32x24
        # self.convlayer2 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu') # 32x32x24
        # self.convlayer3 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu') # 32x32x24

        # self.pooling = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2) # 16x16x24

        # self.normlayer = tf.keras.layers.Normalization(axis=-1,mean=None,invert=False)

        # self.convlayer4 = tf.keras.layers.Conv2D(filters=72, kernel_size=3, padding='same', activation='relu') # 16x16x72
        # self.convlayer5 = tf.keras.layers.Conv2D(filters=72, kernel_size=3, padding='same', activation='relu') # 16x16x72
        # self.convlayer6 = tf.keras.layers.Conv2D(filters=72, kernel_size=3, padding='same', activation='relu') # 16x16x72

        self.global_pool = tf.keras.layers.GlobalAvgPool2D() # 1x1x72

        # self.out = tf.keras.layers.Dense(10, activation='softmax')

        self.loss_function = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()

        self.metrics_list = [
                    tf.keras.metrics.Mean(name="loss"),
                    tf.keras.metrics.BinaryAccuracy(name="acc"), # only for subtask 0, not for subtask 1
                    ]

    @tf.function
    def call(self, x):
        x = self.convlayer1(x) ## trying it out as simple as possible
        # x = self.convlayer2(x)
        # x = self.convlayer3(x)
        # x = self.pooling(x)
        # x = self.convlayer4(x)
        # x = self.convlayer5(x)
        # x = self.convlayer6(x)
        # x = self.global_pool(x)
        x = tf.keras.layers.TimeDistributed(self.global_pool())(x)

        # Once you have encoded all images as vectors, the shape of the tensor should be (batch, sequence-length, features),
        # which can be fed to a non-convolutional standard LSTM.
        return x


    @property
    def metrics(self):
        return self.metrics_list

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    @tf.function
    def train_step(self, input):
        img, label = input

        with tf.GradientTape() as tape:
            prediction = self(img, training=True)
            loss = self.loss_function(label, prediction)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # update loss metric
        self.metrics[0].update_state(loss)

        # for all metrics except loss, update states (accuracy etc.)
        for metric in self.metrics[1:]:
            metric.update_state(label, prediction) # + tf.reduce_sum(self.losses)

        # return a dictionary mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, input):

        img, label = input

        prediction = self(img, training=False)
        loss = self.loss_function(label, prediction) # + tf.reduce_sum(self.losses)

        # update loss metric
        self.metrics[0].update_state(loss)

        # for accuracy metrics:
        for metric in self.metrics[1:]:
            metric.update_state(label, prediction)

        # return a dictionary mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
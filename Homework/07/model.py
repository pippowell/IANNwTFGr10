
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from dataset import shape_ds, sequence_len # shape of the input (batch_size, sequence_length, feature)

# print(shape_ds)

class BasicCNN_LSTM(tf.keras.Model):
    def __init__(self, sequence_length):
        super(BasicCNN_LSTM, self).__init__()

        # input 32x32x3 with 3 as the color channels
        self.convlayer = tf.keras.layers.Conv2D(
            filters=48, kernel_size=3, padding='same', activation='relu', input_shape=shape_ds[2:]) # input_shape(28,28,1)

        self.global_pool = tf.keras.layers.GlobalAvgPool2D()

        self.timedist = tf.keras.layers.TimeDistributed(self.global_pool)()

        # self.out = tf.keras.layers.Dense(10, activation='softmax')

        # lstm ??
        self.lstm = tf.keras.layers.LSTMCell(sequence_length)

        self.rnn = tf.keras.layers.RNN(self.lstm)()

        self.loss_function = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()

        self.metrics_list = [
                    tf.keras.metrics.Mean(name="loss"),
                    tf.keras.metrics.BinaryAccuracy(name="acc"), # only for subtask 0, not for subtask 1
                    ]

    @tf.function
    def call(self, x):
        x = self.convlayer(x)  # trying it out as simple as possible
        #x = self.global_pool(x)
        x = self.timedist(x)
        x = self.rnn(x)

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



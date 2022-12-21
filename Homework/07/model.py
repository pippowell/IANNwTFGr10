
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from dataset import shape_ds, sequence_len # shape of the input (batch_size, sequence_length, feature)

class ourlstm(tf.keras.layers.AbstractRNNCell):

    def __init__(self, units, **kwargs): # units = units of the weight matrixs in each dense layer = units of the output
        super().__init__(**kwargs)

        # self.recurrent_units_1 = recurrent_units_1
        # self.recurrent_units_2 = recurrent_units_2
        
        self.layer1 = tf.keras.layers.Dense(units, activation='sigmoid')
        self.layer2 = tf.keras.layers.Dense(units, activation='sigmoid')
        self.layer3 = tf.keras.layers.Dense(units, activation='tanh')
        self.layer4 = tf.keras.layers.Dense(units, activation='sigmoid')

        # # layer normalization for trainability
        # self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        
        # # layer normalization for trainability
        # self.layer_norm_2 = tf.keras.layers.LayerNormalization()
    
    @property
    def state_size(self):
        return [tf.TensorShape([self.units]), 
                tf.TensorShape([self.units])]
    @property
    def output_size(self):
        return [tf.TensorShape([self.units])]
    
    def get_initial_state(self): #, inputs=None, batch_size=None, dtype=None):
        return [tf.zeros([self.units]), 
                tf.zeros([self.units])]


    # The LSTM-cell layer’s call method should take one (batch of) feature vector(s) as its input, 
    # along with the ”states”, a list containing the different state tensors of the LSTM cell (cell state and hidden state!).
    def call(self, inputs, states):
        # unpack the states
        cell_s = states[0]
        hidden_s = states[1]

        concat_value = tf.concat([inputs, hidden_s], axis=0)

        x1 = self.layer1(concat_value) # correct axis?
        x1 = tf.math.multiply(x1, cell_s)

        x2 = self.layer2(concat_value)
        x3 = self.layer3(concat_value)

        x3 = tf.math.multiply(x2, x3)
        new_cell_s = tf.math.add(x1, x3)

        x4 = self.layer4(concat_value)
        new_hidden_s = tf.math.multiply(new_cell_s, tf.math.tanh(new_cell_s))

        return new_hidden_s, [new_hidden_s, new_cell_s]
        # The returns should be the output of the LSTM, to be used to compute the model
        # output for this time-step (usually the hidden state), as well as a list containing
        # the new states (e.g. [new hidden state, new cell state])

class BasicCNN_LSTM(tf.keras.Model):
    def __init__(self):
        super(BasicCNN_LSTM, self).__init__()

        # input 32x32x3 with 3 as the color channels
        self.convlayer = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu', input_shape=shape_ds[2:]) # input_shape(28,28,1)

        self.global_pool = tf.keras.layers.GlobalAvgPool2D()
        self.timedist = tf.keras.layers.TimeDistributed(self.global_pool)#()
        # input of lstm: (bs, sequencelen, feature(output of cnn))

        # implementing lstm manually ?? - create tf layer (backprop is done by tf)
        # self.lstm = tf.keras.layers.LSTMCell(sequence_length)

        self.rnn = tf.keras.layers.RNN(ourlstm(32)) 

        self.loss_function = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()

        self.metrics_list = [
                    tf.keras.metrics.Mean(name="loss"),
                    tf.keras.metrics.BinaryAccuracy(name="acc"), # only for subtask 0, not for subtask 1
                    ]

    @tf.function
    def call(self, x):
        x = self.convlayer(x)  # trying it out as simple as possible
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

        # for all metrics 
        for metric in self.metrics:
            metric.update_state(label, prediction) # + tf.reduce_sum(self.losses)

        # return a dictionary mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, input):

        img, label = input

        prediction = self(img, training=False)
        loss = self.loss_function(label, prediction) # + tf.reduce_sum(self.losses)

        # for all metrics 
        for metric in self.metrics:
            metric.update_state(label, prediction) # + tf.reduce_sum(self.losses)

        # return a dictionary mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
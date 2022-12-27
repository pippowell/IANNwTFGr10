
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from dataset import shape_ds, sequence_len # shape of the input (batch_size, sequence_length, feature)

class ourlstm(tf.keras.layers.AbstractRNNCell):

    def __init__(self, units, **kwargs): # units = units of the weight matrixs in each dense layer = units of the output
        super().__init__(**kwargs)

        self.units = units

        self.forgetgate = tf.keras.layers.Dense(units, 
                                                kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None), 
                                                activation='sigmoid')
        self.inputgate1 = tf.keras.layers.Dense(units, 
                                                kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None), 
                                                activation='sigmoid')
        self.inputgate2 = tf.keras.layers.Dense(units, 
                                                kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None), 
                                                activation='tanh')
        self.outputgate = tf.keras.layers.Dense(units, 
                                                kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None), 
                                                activation='sigmoid')
    
    @property
    def state_size(self):
        return [tf.TensorShape([self.units]), 
                tf.TensorShape([self.units])]
    @property
    def output_size(self):
        return [tf.TensorShape([self.units])]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [tf.zeros([batch_size, self.units]), 
                tf.zeros([batch_size, self.units])]


    # The LSTM-cell layer’s call method should take one (batch of) feature vector(s) as its input, 
    # along with the ”states”, a list containing the different state tensors of the LSTM cell (cell state and hidden state!).
    def call(self, inputs, states):
        # unpack the states
        cell_s = states[0]
        hidden_s = states[1]

        concat_value = tf.concat([inputs, hidden_s], axis=-1) # Leon: or use a tuple

        x1 = self.forgetgate(concat_value)
        x1 = tf.math.multiply(x1, cell_s) # or use *

        x2 = self.inputgate1(concat_value)
        x3 = self.inputgate2(concat_value)

        x3 = tf.math.multiply(x2, x3)
        new_cell_s = tf.math.add(x1, x3)

        x4 = self.outputgate(concat_value)
        new_hidden_s = tf.math.multiply(x4, tf.math.tanh(new_cell_s))

        return new_hidden_s, (new_hidden_s, new_cell_s)
        # The returns should be the output of the LSTM, to be used to compute the model
        # output for this time-step (usually the hidden state), as well as a list containing
        # the new states (e.g. [new hidden state, new cell state])

class BasicCNN_LSTM(tf.keras.Model):
    def __init__(self):
        super(BasicCNN_LSTM, self).__init__()

        self.convlayer1 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu')#, input_shape=shape_ds[2:])
        self.convlayer2 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu')#, input_shape=shape_ds[2:])
        self.convlayer3 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu')#, input_shape=shape_ds[2:])
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        self.global_pool = tf.keras.layers.GlobalAvgPool2D()
        self.timedist = tf.keras.layers.TimeDistributed(self.global_pool)

        self.rnn = tf.keras.layers.RNN(ourlstm(8), return_sequences=True) 
        self.batchnorm2 = tf.keras.layers.BatchNormalization()

        self.outputlayer = tf.keras.layers.Dense(units=1, activation=None) 
        
        # self.loss_function = tf.keras.losses.MeanSquaredError()
        # self.optimizer = tf.keras.optimizers.Adam()

        # self.metrics_list = [
        #             tf.keras.metrics.Mean(name="loss"),
        #             tf.keras.metrics.MeanAbsoluteError(name="acc")
        #             ]

    # @tf.function # remove when debugging
    def call(self, x):
        print(f"initial shape: {x.shape}")
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.convlayer3(x)
        print(f"shape after cnn: {x.shape}")
        x = self.batchnorm1(x)
        x = self.timedist(x) # after this the shape.x should be (bs, sequence-length, features) before LSTM
        print(f"shape after timedist&pooling: {x.shape}")
        
        x = self.rnn(x)
        print(f"shape after rnn: {x.shape}")
        x = self.batchnorm2(x)
        x = self.outputlayer(x)
        print(f"shape after output: {x.shape}")

        return x

testmodel = BasicCNN_LSTM()
# testmodel.summary()

    # @property
    # def metrics(self):
    #     return self.metrics_list

    # def reset_metrics(self):
    #     for metric in self.metrics:
    #         metric.reset_states()

    # # @tf.function
    # def train_step(self, input):
    #     img, label = input

    #     with tf.GradientTape() as tape:
    #         prediction = self(img, training=True)
    #         loss = self.loss_function(label, prediction)

    #     gradients = tape.gradient(loss, self.trainable_variables)
    #     self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    #     # metrics update
    #     self.metrics[0].update_state(loss)
    #     self.metrics[1].update_state(label, prediction)

    #     # return a dictionary mapping metric names to current value
    #     return {m.name: m.result() for m in self.metrics}

    # # @tf.function
    # def test_step(self, input):

    #     img, label = input

    #     prediction = self(img, training=False)
    #     loss = self.loss_function(label, prediction) # + tf.reduce_sum(self.losses)

    #     # metrics update
    #     self.metrics[0].update_state(loss)
    #     self.metrics[1].update_state(label, prediction)

    #     # return a dictionary mapping metric names to current value
    #     return {m.name: m.result() for m in self.metrics}

# output is (32,6,1) and the target is (32,6) -> squeeze the dimension of output, expand the target

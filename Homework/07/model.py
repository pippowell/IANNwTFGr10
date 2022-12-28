
import tensorflow as tf
import dataset 

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

    def get_config(self):
        return {"hidden_units": self.units}
                

class BasicCNN_LSTM(tf.keras.Model):
    def __init__(self):
        super(BasicCNN_LSTM, self).__init__()

        self.convlayer1 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu', 
                                                batch_input_shape=(dataset.batch_size, dataset.sequence_len, 28, 28, 1))#, input_shape=(28, 28, 1))
        self.convlayer2 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu', 
                                                batch_input_shape=(dataset.batch_size, dataset.sequence_len, 28, 28, 1))#, input_shape=(28, 28, 1))
        self.convlayer3 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu', 
                                                batch_input_shape=(dataset.batch_size, dataset.sequence_len, 28, 28, 1))#, input_shape=(28, 28, 1))
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        self.global_pool = tf.keras.layers.GlobalAvgPool2D()
        self.timedist = tf.keras.layers.TimeDistributed(self.global_pool)

        self.rnn = tf.keras.layers.RNN(ourlstm(8), return_sequences=True) 
        self.batchnorm2 = tf.keras.layers.BatchNormalization()

        self.outputlayer = tf.keras.layers.Dense(units=1, activation=None) 
        
    @tf.function # Leon: comment it out when debugging
    def call(self, x):
        # print(f"initial shape: {x.shape}")
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.convlayer3(x)
        # print(f"shape after cnn: {x.shape}")
        
        x = self.batchnorm1(x)
        x = self.timedist(x) 
        # print(f"shape after timedist&pooling: {x.shape}") # shape should be (bs, sequence-length, features) before LSTM

        x = self.rnn(x)
        # print(f"shape after rnn: {x.shape}")
        
        x = self.batchnorm2(x)
        x = self.outputlayer(x)
        # print(f"shape after output: {x.shape}")

        return x

# # testing
# testmodel = ourlstm(10)
# print(testmodel.get_config())
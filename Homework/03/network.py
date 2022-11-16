import mnisttf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense
import tensorflow as tf

# 2.3 Building a deep neural network with TensorFlow

# 1st approach
class Dense(tf.keras.layers.Layer):

    def __init__(self, n_units, activation_function, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.n_units = n_units
        self.activation_function = activation_function
        
    def build(self, input_shape):
        self.w = tf.Variable(tf.random.normal([input_shape[-1], self.n_units]), name='weights')
        self.b = tf.Variable(tf.zeros([self.n_units]), name='bias')

    def call(self, input):
        x = input @ self.w + self.b
        
        return self.activation_function(x)

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        
        # good (albeit arbitrary) starting point would be to have two hidden layers with 256 units each
        self.hidden_layer_1 = Dense(n_units=256, activation_function=tf.nn.relu)
        self.hidden_layer_2 = Dense(n_units=256, activation_function=tf.nn.relu)

        # softmax as activation function returns a probability distribution of length 10
        self.output_layer = Dense(n_units=10, activation_function=tf.nn.softmax)
    
    @tf.function
    def __call__(self, input):
        
        output_1 = self.hidden_layer_1(input)
        output_2 = self.hidden_layer_2(output_1)
        output = self.output_layer(output_2)
        
        return output

# 2nd approach (not done yet)
class MyModule(tf.Module):
    def __init__(self, input_dim, output_dim, name=None):
        super().__init(name=name)
     
from tensorflow.keras.layers import Dense
import tensorflow as tf

# 2.3 Building a deep neural network with TensorFlow

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        
        # good (albeit arbitrary) starting point would be to have two hidden layers with 256 units each
        self.hidden_layer_1 = Dense(units=256, activation=tf.nn.relu)
        self.hidden_layer_2 = Dense(units=256, activation=tf.nn.relu)

        # softmax as activation function returns a probability distribution of length 10
        self.output_layer = Dense(units=10, activation=tf.nn.softmax)
    
    @tf.function
    def __call__(self, input):
        
        x = self.hidden_layer_1(input)
        x = self.hidden_layer_2(x)
        x = self.output_layer(x)
        
        return x
     
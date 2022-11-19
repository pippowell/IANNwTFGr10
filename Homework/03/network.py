from tensorflow.keras.layers import Dense
import tensorflow as tf

# 2.3 Building a deep neural network with TensorFlow
class MyModel(tf.keras.Model):
    def __init__(self):

        '''
        creates a neural network model with 2 hidden layers and an output layer
        '''

        super(MyModel, self).__init__()
        
        # create 2 hidden layers with 256 units and ReLU as the activation function
        self.hidden_layer_1 = Dense(units=256, activation=tf.nn.relu)
        self.hidden_layer_2 = Dense(units=256, activation=tf.nn.relu)

        # create an output layer with 10 units and softmax as the activation function (prob distribution of length 10)
        self.output_layer = Dense(units=10, activation=tf.nn.softmax)
    
    @tf.function
    def __call__(self, input):

        '''
        :param input: input to the network (tensor)
        :return: output of final layer
        '''

        x = self.hidden_layer_1(input)
        x = self.hidden_layer_2(x)
        x = self.output_layer(x)
        
        return x
     
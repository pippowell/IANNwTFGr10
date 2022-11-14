import tensorflow as tf
from tensorflow.keras.layers import Dense

class Network(tf.keras.Model):
    def _init_(self,n_layers):
        super(Network,self)._init_()
        layers = []
        for i in range(n_layers+1):
            self.dense = tf.keras.layers.Dense(256,activation=tf.nn.relu)
            self.layers.append(self.dense)
        self.out = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    @tf.function
    def call(self,input):
        x = self.layers[0](input)
        for i in range(len(layers)+1):
            x = self.layers[i+1](x)
        return x

def train(epoch, model, traindata, testdata, lossfunction, optimizer):





import tensorflow as tf
from tensorflow.keras.layers import Dense

class Network(tf.keras.Model):
    def _init_(self,n_layers):
        super(Network,self)._init_()
        self.layers = []
        for i in range(n_layers+1):
            self.dense = Dense(256,activation=tf.nn.relu)
            self.layers.append(self.dense)
        self.out = Dense(10, activation=tf.nn.softmax)

    def call(self,input):
        x = self.layers[0](input)
        for i in range(len(self.layers)+1):
            x = self.layers[i+1](x)
        return x





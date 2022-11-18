import tensorflow as tf
from tensorflow.keras.layers import Dense

class Network(tf.keras.Model):

    def cyclelayers(self,cutoff,i):
        tf.less(i,cutoff)

    def addlayer(self,i):
        layer = Dense(256,activation=tf.nn.relu)
        self.layers.append(layer)
        i++1
        return (self.layers,i)

    def dyninput(self,i,x):
        x = self.layers[i+1](x)
        i++1
        return (x,i)

    def _init_(self,n_layers):
        super(Network,self,n_layers)._init_()
        self.n_layers = n_layers
        self.layers = []
        i = 0
        tf.while_loop(self.cyclelayers(self.n_layers,i),self.addlayer(i))
        self.out = Dense(10, activation=tf.nn.softmax)

    def call(self,input):
        x = self.layers[0](input)
        i = 0
        tf.while_loop(self.cyclelayers(self.n_layers,i),self.dyninput(i,x))
        x = self.out(x)
        return x





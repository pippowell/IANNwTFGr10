import tensorflow as tf
from tensorflow.keras.layers import Dense

# This was an attempt at creating a model class which would allow dynamic (user-defined) number of layers during model creation. It does not function. network.py
# contains our actual solution to the network creation part of this week's task

class Network(tf.keras.Model):

    # defining conditions and bodies for the tf while loops since Python for loops do not work here
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

    # initialize the model with the given number of layers
    def _init_(self,n_layers):
        super(Network,self,n_layers)._init_()
        self.n_layers = n_layers
        self.layers = []
        i = 0
        tf.while_loop(self.cyclelayers(self.n_layers,i),self.addlayer(i))
        self.out = Dense(10, activation=tf.nn.softmax)

    # pass the input to the model and propagate activity through the network, returning the final output
    def call(self,input):
        x = self.layers[0](input)
        i = 0
        tf.while_loop(self.cyclelayers(self.n_layers,i),self.dyninput(i,x))
        x = self.out(x)
        return x





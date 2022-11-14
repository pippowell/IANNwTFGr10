import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np

#2.1 Load Dataset
(train_ds, test_ds), ds_info = tfds.load ('mnist', split =['train', 'test'], as_supervised = True, with_info = True)

# print("ds_info: \n", ds_info)

#Dataset Info
#Number of Training Images: 60000
#Number of Testing Images: 10000
#Image Shape: 28,28,1 (28 by 28 images, grayscale)
#Range of Pixels: 0-255 (grayscale)

# tfds.show_examples(train_ds, ds_info)

#2.2 Setting up the data pipeline
def prepare_data(dataset):

    # convert data from uint8 to float32
    mnist = mnist.map(lambda img, target: (tf.cast(img, tf.float32), target))

    # flatten the images into vectors
    dataset = dataset.map(lambda img, target: (tf.reshape(img, (-1,)), target))

    # input normalization, just bringing image values from range [0, 255] to [-1, 1]
    dataset = dataset.map(lambda img, target: ((img / 128.) - 1., target))

    # create one-hot targets
    dataset = dataset.map(lambda img, target: (img, tf.one_hot(target, depth=10)))

    # cache
    dataset = dataset.cache()

    # shuffle, batch, prefetch
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(20)

    # return preprocessed dataset
    return dataset

# 2.3 Building a deep neural network with TensorFlow
class Dense(tf.keras.layers.Layer):

    def __init__(self, n_units, activation_function, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.n_units = n_units
        self.activation_function = activation_function
        
    def build(self, input_shape):
        self.w = tf.Variable(tf.random.normal([input_shape[-1], self.n_units]), name='weights')
        self.b = tf.Variable(tf.zeros([self.n_units]), name='bias')

    def call(self, inputs):
        x = inputs @ self.w + self.b
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
    def __call__(self, inputs):
        
        output_1 = self.hidden_layer_1(inputs)
        output_2 = self.hidden_layer_2(output_1)
        output = self.output_layer(output_2)
        
        return output

# 2.4 Training the network

# hyperparameters
epoch = 10
learning_rate = 0.1

# define training and test datasets
train_dataset = train_ds.apply(prepare_data)
test_dataset = test_ds.apply(prepare_data)

# choose optimizer and loss
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0)
loss_function = tf.keras.losses.MeanSquaredError()

# Different arrays for the different values for visualization
train_losses = []
test_losses = []
test_accuracies = []

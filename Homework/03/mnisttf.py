import tensorflow_datasets as tfds
import tensorflow as tf

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
def prepare_mnist_data(mnist): 
    
    # convert unint8 to tf.float
    mnist = mnist.map(lambda img, target: (tf.cast(img, tf.float32), target))

    # flatten the image to (28, 28)
    mnist = mnist.map(lambda img, target: (tf.reshape(img, (-1,)), target))

    # normalize the input
    mnist = mnist.map(lambda img, target: ((img/128)-1, target))

    # encode the labels as one-hot vectors
    mnist = mnist.map(lambda img, target: (img, tf.one_hot(target, depth = 10)))

    return mnist

print("prepare train_ds", train_ds.apply(prepare_mnist_data))
print("prepare test_ds", test_ds.apply(prepare_mnist_data))

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

# good (albeit arbitrary) starting point would be to have two hidden layers with 256 units each
hidden_layer_1 = Dense(n_units=256, activation_function=tf.nn.relu)
hidden_layer_2 = Dense(n_units=256, activation_function=tf.nn.relu)
output_layer = Dense(n_units=9, activation_function=tf.nn.softmax)

# 2.4 Training the network




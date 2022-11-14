import tensorflow_datasets as tfds
import tensorflow as tf

#2.1 Load Dataset
(train_ds, test_ds),ds_info = tfds.load ('mnist', split =['train', 'test'], as_supervised = True, with_info = True)

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
    mnist = mnist.map(lambda img, target: (tf.cas(img, tf.float32), target))

    # flatten the image to (28, 28)
    mnist = mnist.map(lambda img, target: (tf.reshape(img, (-1)), target))

    # normalize the input
    mnist = mnist.map(lambda img, target: ((img/128)-1, target))

    # encode the labels as one-hot vectors
    mnist = mnist.map(lambda img, target: (img, tf.one_hot(target, depth = 10)))

# 2.3 Building a deep neural network with TensorFlow

    


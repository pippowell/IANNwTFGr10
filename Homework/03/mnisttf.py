import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

#2.1 Load Dataset
(train_ds,test_ds),ds_info = tfds.load ('mnist', split =[ 'train','test'], as_supervised = True, with_info = True)

print(ds_info)

#Dataset Info
#Number of Training Images: 60000
#Number of Testing Images: 10000
#Image Shape: 28,28,1 (28 by 28 images, grayscale)
#Range of Pixels: 0-255 (grayscale)

tfds.show_examples(train_ds,ds_info)

#2.2 Data Pipeline

def prepare_data(dataset):
    # convert data from uint8 to float32
    dataset = dataset.map(lambda img, target: (tf.cast(img, tf.float32), target))

    #flatten the images into vectors
    dataset = dataset.map(lambda img, target: (tf.reshape(img, (-1,)), target))

    #input normalization, just bringing image values from range [0, 255] to [-1, 1]
    dataset = dataset.map(lambda img, target: ((img / 128.) - 1., target))

    #create one-hot targets
    dataset = dataset.map(lambda img, target: (img, tf.one_hot(target, depth=10)))

    # cache
    dataset = dataset.cache()

    # shuffle, batch, prefetch
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(20)

    # return preprocessed dataset
    return dataset

train_dataset = train_ds.apply(prepare_data)
test_dataset = test_ds.apply(prepare_data)
import tensorflow_datasets as tfds
import tensorflow as tf
# import numpy as np


# Load the dataset and construct a tf.Data.Dateset for testing and training using the images only
# (img_train, _), (img_test, _) = tf.keras.datasets.mnist.load_data()

train_ds, test_ds = tfds.load('mnist', split=['train', 'test'], as_supervised=True)
print(f"train_ds: {train_ds}") # shape=(28, 28, 1)
print(f"test_ds: {test_ds}") # shape=(28, 28, 1)

# Introduce a hyperparameter to control how noisy your data will be
mean = 0.5
stddev = 0.5
batchsize = 32

def preprocess(dataset):

    # Normalize the images and make sure that you have sensible dimensions
    dataset = dataset.map(lambda img, target: (tf.cast(img, tf.float32), target))
    dataset = dataset.map(lambda img, target: ((img / 128.) - 1., target)) # shape=(28, 28, 1)
    # print(f"after norm: {dataset}")

    # Add a third dimension
    dataset =  dataset.map(lambda img, target: (tf.expand_dims(img, axis=-1, name=None), target)) # shape=(28, 28, 1, 1)
    # print(f"after dimension expansion: {dataset}")

    # get img shape before adding noise to image
    for img, _ in dataset.take(1):
        img_shape = img.shape

    # Add noise to the input image
    noise = tf.random.normal(shape=img_shape, mean=mean, stddev=stddev, dtype=tf.dtypes.float32) # noise.shape = (28, 28, 1)
    # print(f"noise: {noise.shape}")
    dataset = dataset.map(lambda img, target: (tf.clip_by_value(img + noise, clip_value_min=-1, clip_value_max=1), target)) # shape=(28, 28, 28, 1)

    # Cache, shuffle, batch, prefetch
    dataset = dataset.cache()
    dataset = dataset.shuffle(1000)
    # print(f"before batch: {dataset}")
    dataset = dataset.batch(batchsize) # shape=(None, 28, 28, 28, 1)
    # print(f"after batch: {dataset}")
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

train_ds = preprocess(train_ds) # shape=(None, 28, 28, 28, 1)
test_ds = preprocess(test_ds) # shape=(None, 28, 28, 28, 1)

print(f"preprocessed train_ds: {train_ds}") 
print(f"preprocessed test_ds: {test_ds}")
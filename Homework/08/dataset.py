import tensorflow_datasets as tfds
import tensorflow as tf
# import numpy as np


# Load the dataset and construct a tf.Data.Dateset for testing and training using the images only
# (img_train, _), (img_test, _) = tf.keras.datasets.mnist.load_data()

train_ds, test_ds = tfds.load('mnist', split=['train', 'test'], as_supervised=True)
print(f"train_ds: {train_ds}")
print(f"val_ds: {test_ds}")

# Introduce a hyperparameter to control how noisy your data will be
mean = 0.5
stddev = 0.5
batchsize = 32

def preprocess(dataset):

    # Normalize the images and make sure that you have sensible dimensions
    dataset = dataset.map(lambda img, target: (tf.cast(img, tf.float32), target))
    dataset = dataset.map(lambda img, target: ((img / 128.) - 1., target))
    print(f"after norm: {dataset}")

    # Add a third dimension
    dataset =  dataset.map(lambda img, target: (tf.expand_dims(img, axis=-1, name=None), target))
    print(f"after dimension expansion: {dataset}")

    # Add noise to the input image
    noise = dataset.map(lambda img, target: (tf.random.normal(shape=img.shape, mean=mean, stddev=stddev), target))
    noisyds = dataset.map(lambda img, target: (tf.clip_by_value(img + noise, clip_value_min=-1, clip_value_max=1), target))

    # Cache, shuffle, batch, prefetch
    dataset = dataset.cache()
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batchsize)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return noisyds

img_train, _ = preprocess(train_ds)
img_test, _ = preprocess(test_ds)

print(f"preprocessed img_train: {img_train}")
print(f"preprocessed img_test: {img_test}")
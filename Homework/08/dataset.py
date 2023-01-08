import tensorflow_datasets as tfds
import tensorflow as tf
# import numpy as np
import matplotlib.pyplot as plt


# Load the dataset and construct a tf.Data.Dateset for testing and training using the images only
# (img_train, _), (img_test, _) = tf.keras.datasets.mnist.load_data()

train_ds, test_ds = tfds.load('mnist', split=['train', 'test'], as_supervised=True)
# print(f"train_ds: {train_ds}") # shape=(28, 28, 1)
# print(f"test_ds: {test_ds}") # shape=(28, 28, 1)

# Introduce a hyperparameter to control how noisy your data will be
mean = 0.5
stddev = 0.5
batchsize = 32

def preprocess(dataset):

    # Normalize the images and make sure that you have sensible dimensions
    dataset = dataset.map(lambda img, _: (tf.cast(img, tf.float32), tf.cast(img, tf.float32)))
    dataset = dataset.map(lambda img, target_img: ((img / 128.) - 1., (target_img / 128.) - 1)) # shape=(28, 28, 1)
    # print(f"after norm: {dataset}")

    # Add a third dimension
    # dataset =  dataset.map(lambda img, target_img: (tf.expand_dims(img, axis=-1, name=None), target_img)) # shape=(28, 28, 1, 1)
    # print(f"after dimension expansion: {dataset}")

    # get img shape before adding noise to image
    for img, _ in dataset.take(1):
        img_shape = img.shape

    noise_factor = 0.5
    # create noise to be added to image
    noise = noise_factor*tf.random.normal(shape=img_shape, mean=mean, stddev=stddev, dtype=tf.dtypes.float32)
    # Add noise to the input image
    dataset = dataset.map(lambda img, target_img: (img + noise, target_img))
    dataset = dataset.map(lambda img, target_img: (tf.clip_by_value(img, clip_value_min=-1, clip_value_max=1), target_img)) # shape=(28, 28, 1, 1)
    # print(f"after noise: {dataset}")

    # Cache, shuffle, batch, prefetch
    dataset = dataset.cache()
    dataset = dataset.shuffle(1000)
    # print(f"before batch: {dataset}")

    dataset = dataset.batch(batchsize) # shape=(None, 28, 28, 1)
    # print(f"after batch: {dataset}")
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

noisy_train_ds = preprocess(train_ds) # shape=(None, 28, 28, 1, 1)
noisy_test_ds = preprocess(test_ds) # shape=(None, 28, 28, 1, 1)

# Check the shape
# for noisy, original in noisy_train_ds.take(1):
    # print(noisy.shape)
    # print(original.shape)

# to visualize the input and noisy target image
# Get the first element in the dataset
noisy_image, original_image = next(iter((noisy_train_ds.take(1))))

# Display the noisy image
# plt.imshow(noisy_image[0])
# plt.show()

# # Display the original image
# plt.imshow(original_image[0])
# plt.show()

# print(f"noisy_train_ds: {noisy_train_ds}") # shape=(28, 28, 1)
# print(f"noisy_test_ds: {noisy_test_ds}") # shape=(28, 28, 1)

# print(f"preprocessed train_ds: {noisy_train_ds}") 
# print(f"preprocessed test_ds: {noisy_test_ds}")

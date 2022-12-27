import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np


# 2.1 Load Dataset
(train_ds, val_ds), ds_info = tfds.load ('mnist', split =['train', 'test'], as_supervised = True, with_info = True)

# print("ds_info: \n", ds_info)
# tfds.show_examples(train_ds, ds_info)

sequence_len = 6

def preprocess(dataset, batchsize, sequence_len):
    '''
    :param dataset: the dataset to be prepared for input into the network
    :param batchsize: the desired batchsize
    :param sequence_len: sequence the dataset (you may think this as a another batch)
    :return: sequenced dataset with sequential target
    '''

    # Step 1 - General Preprocessing

    # convert data from uint8 to float32
    dataset = dataset.map(lambda img, target: (tf.cast(img, tf.float32), target))

    # input normalization, just bringing image values from range [0, 255] to [-1, 1]
    dataset = dataset.map(lambda img, target: ((img / 128.) - 1., target))

    # print(dataset.shape)
    # The output of that lambda function should be a tuple of two tensors of shapes (num_images, height, width, 1) and (num_images, 1) or (num_images,)

    # Step 3 - Sequence Batching & Batching & Prefetching
    dataset = dataset.batch(sequence_len)

    # change the target

    # alternate positive, negative target values
    range_vals = tf.range(sequence_len)

    dataset = dataset.map(lambda img, target:
                          (img, tf.where(tf.math.floormod(range_vals,2)==0, target, -target)))

    dataset = dataset.map(lambda img, target:
                          (img, (tf.math.cumsum(target))))

    # cache
    dataset = dataset.cache()

    # shuffle, batch, prefetch
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batchsize)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # for _, label in dataset:
    #     label = np.expand_dims(label, axis=-1)

    # The shape of your tensors should be (batch, sequence-length, features).
    return dataset

# Test

train_ds = preprocess(train_ds, 64, 6)
val_ds = preprocess(val_ds, 64, 6)


for img, label in train_ds.take(1):
    shape_ds = img.shape

    print(img.shape, label.shape) 
    # (64, 6, 28, 28, 1)                 (64, 6)
    # (bs, num_images, height, width, 1) (bs, num_images, 1)

    ## Wooki: is the issue here that lable.shape =! (64,6,1)??
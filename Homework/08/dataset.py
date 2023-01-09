import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

train_ds, test_ds = tfds.load('mnist', split=['train', 'test'], as_supervised=True) # each the shape of (28, 28, 1), 1)

# Introduce a hyperparameter to control how noisy your data will be
mean = 0.5
stddev = 0.5
batchsize = 32

def preprocess(dataset):

    # Normalize the images and make sure that you have sensible dimensions
    dataset = dataset.map(lambda img, label: (tf.cast(img, tf.float32), tf.cast(img, tf.float32),label))
    dataset = dataset.map(lambda img, target_img, label: ((img / 128.) - 1., (target_img / 128.) - 1,label)) # shape=(28, 28, 1)

    # get img shape before adding noise to image
    for img, _, _ in dataset.take(1):
        img_shape = img.shape

    # Add noise to the input image
    noise_factor = 0.5
    noise = noise_factor*tf.random.normal(shape=img_shape, mean=mean, stddev=stddev, dtype=tf.dtypes.float32)
    dataset = dataset.map(lambda img, target_img, label: (img + noise, target_img, label))
    dataset = dataset.map(lambda img, target_img, label: (tf.clip_by_value(img, clip_value_min=-1, clip_value_max=1), target_img, label))

    # Cache, shuffle, batch, prefetch
    dataset = dataset.cache()
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batchsize) 

    # take label info to new dataset
    labels = dataset.map(lambda _, __, label: (label))
    noisy_img = dataset.map(lambda img, _, __: (img))
    target_img = dataset.map(lambda _, target_img, __: target_img)

    # remove label info from main dataset
    dataset = dataset.map(lambda img, target_img, _:(img, target_img))

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, labels

noisy_train_ds, trainlabels = preprocess(train_ds) # shape=(None, 28, 28, 1), (None, 28, 28, 1), (None)
noisy_test_ds, testlabels = preprocess(test_ds) # shape=(None, 28, 28, 1), (None, 28, 28, 1), (None)

# noisy_img_train, target_img_train, labels_train = preprocess(train_ds)
# noisy_img_test, target_img_test, labels_test = preprocess(train_ds)

# to visualize the input and noisy target image
# Get the first element in the dataset
# noisy_image, original_image = next(iter((noisy_train_ds.take(1))))

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

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

#2.2 Data Pipeline



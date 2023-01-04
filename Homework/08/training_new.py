import tensorflow as tf
import dataset_new
import model
import matplotlib.pyplot as plt
import datetime as datetime
from pathlib import Path
import numpy as np

# Initiate epochs and learning rate as global variables
epochs = 2 # 10
lr = 1e-3

ae = model.autoencoder()

loss = tf.keras.losses.MSE()
opti = tf.keras.optimizers.Adam(learning_rate=lr)

ae.compile(loss=loss, optimizer=opti)

# noisy_train_x, _ = dataset.noisy_train_ds
# train_x, _ = dataset.train_ds

# noisy_test_x, _ = dataset.noisy_test_ds
# test_x, _ = dataset.test_ds

# noisy_train_x = list()
# noisy_test_x = list()
# train_x = list()
# test_x = list()

# for x, _ in dataset.noisy_train_ds: noisy_train_x.append(x)
# for x, _ in dataset.noisy_test_ds: noisy_test_x.append(x)
# for x, _ in dataset.train_ds: train_x.append(x)
# for x, _ in dataset.test_ds: test_x.append(x)

# print(f"noisy_train_x: {len(noisy_train_x)}")
# print(f"train_x: {len(train_x)}")
# print(f"noisy_test_x: {len(noisy_test_x)}")
# print(f"test_x: {len(test_x)}")

ae.fit(dataset_new.noisy_train_ds, validation_data=dataset_new.noisy_test_ds, epochs=epochs)
# this part not working!!

print("training done")
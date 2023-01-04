import tensorflow as tf
import dataset 
import model
import matplotlib.pyplot as plt
import datetime as datetime
from pathlib import Path

# Initiate epochs and learning rate as global variables
epochs = 2 # 10
lr = 1e-3

ae = model.autoencoder()

loss = tf.keras.losses.BinaryCrossentropy()
opti = tf.keras.optimizers.Adam(learning_rate=lr)

ae.compile(loss=loss, optimizer=opti)

# noisy_train_x, _ = dataset.noisy_train_ds
# train_x, _ = dataset.train_ds

# noisy_test_x, _ = dataset.noisy_test_ds
# test_x, _ = dataset.test_ds

noisy_train_x = list()
train_x = list()
noisy_test_x = list()
test_x = list()

for x, _ in dataset.noisy_train_ds: noisy_train_x.append(x)
for x, _ in dataset.train_ds: train_x.append(x)

for x, _ in dataset.noisy_test_ds: noisy_test_x.append(x)
for x, _ in dataset.test_ds: test_x.append(x)
    
ae.fit(x=noisy_train_x, y=train_x, validation_data=(noisy_test_x, test_x), epochs=epochs)
print("training done")
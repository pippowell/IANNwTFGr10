import tensorflow as tf
import dataset 
import model
import matplotlib.pyplot as plt
import datetime as datetime
from pathlib import Path

# Initiate epochs and learning rate as global variables
epochs = 10
lr = 1e-3

ae = model.autoencoder()

loss = tf.keras.losses.BinaryCrossentropy()
opti = tf.keras.optimizers.Adam(learning_rate=lr)

ae.compile(loss=loss, optimizer=opti)

ae.fit(x=dataset.noisy_train_ds, y=dataset.train_ds, epochs=epochs, validation_data=(dataset.noisy_test_ds, dataset.test_ds))

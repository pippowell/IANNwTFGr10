import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import dataset 
import model
import matplotlib.pyplot as plt


mymodel = model.BasicCNN_LSTM(dataset.sequence_len)
mymodel.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer="adam")
original = mymodel.fit(dataset.train_dataset, validation_data=dataset.test_dataset, epochs=2)

fig, ax0 = plt.subplots(1, 1, figsize=(8, 10))

ax0.set_title("original")
ax0.plot(original.history["total_frobenius_norm"]/np.max(original.history["total_frobenius_norm"]) * np.max(original.history["val_loss"]))
ax0.plot(original.history["val_loss"])
ax0.plot(original.history["loss"])
ax0.legend(labels=["Total Frobenius Norm", "Validation Loss", "Loss"])
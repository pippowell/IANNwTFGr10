import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import dataset 
import model
import matplotlib.pyplot as plt
import datetime as datetime


# Initiate the logs and metrics
config_name= "HW07"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

train_log_path = f"logs/{config_name}/{current_time}/train"
val_log_path = f"logs/{config_name}/{current_time}/val"

# log writer for training metrics
train_summary_writer = tf.summary.create_file_writer(train_log_path)

# log writer for validation metrics
val_summary_writer = tf.summary.create_file_writer(val_log_path)


# Initiate epochs and learning rate as global variables
epochs = 2 #15
learning_rate = 0.05

# Define arrays for saving values for later visualization
train_forb_norm = []
train_losses = []
train_accuracies = []

val_forb_norm = [] 
val_losses = []
val_accuracies = []

mymodel = model.BasicCNN_LSTM()
mymodel.compile(run_eagerly=True) #loss=tf.keras.losses.MeanSquaredError(), optimizer="adam")
original = mymodel.fit(dataset.train_ds, validation_data=dataset.val_ds, epochs=epochs)

# use eager 

fig, ax0 = plt.subplots(1, 1, figsize=(8, 10))

ax0.set_title("original")
ax0.plot(original.history["total_frobenius_norm"]/np.max(original.history["total_frobenius_norm"]) * np.max(original.history["val_loss"]))
ax0.plot(original.history["val_loss"])
ax0.plot(original.history["loss"])
ax0.legend(labels=["Total Frobenius Norm", "Validation Loss", "Loss"])


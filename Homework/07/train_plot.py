import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import dataset 
import model
import matplotlib.pyplot as plt
import datetime as datetime

model = model.BasicCNN_LSTM()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)#1e-3)
loss = tf.keras.losses.MeanSquaredError()

# compile the model (here, adding a loss function and an optimizer)
model.compile(optimizer = optimizer, loss=loss)

EXPERIMENT_NAME = "CNN_LSTM"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logging_callback = tf.keras.callbacks.TensorBoard(log_dir=f"./logs/{EXPERIMENT_NAME}/{current_time}")

history = model.fit(dataset.train_ds,
                    validation_data=dataset.val_ds,
                    initial_epoch=1,#25,
                    epochs=2,#50,
                    callbacks=[logging_callback])

# save the complete model (incl. optimizer state, loss function, metrics etc.)                
model.save("saved_model")

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
# plt.plot(history.history["acc"])
# plt.plot(history.history["val_acc"])
plt.legend(labels=["training","validation"])
plt.xlabel("Epoch")
plt.ylabel("MSE/MAD")
plt.show()
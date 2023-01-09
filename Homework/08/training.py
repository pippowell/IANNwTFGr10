import tensorflow as tf
import dataset
import model
import matplotlib.pyplot as plt
import datetime as datetime
from pathlib import Path

# Initiate epochs and learning rate as global variables
epochs = 1
lr = 0.99

ae = model.autoencoder(vae=False)
vae = model.autoencoder(vae=True)

loss = tf.keras.losses.MeanSquaredError()
opti = tf.keras.optimizers.Adam(learning_rate=lr)

# save logs with Tensorboard
EXPERIMENT_NAME = "CNN_LSTM"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logging_callback = tf.keras.callbacks.TensorBoard(log_dir=f".Homework/08/logs/{EXPERIMENT_NAME}/{current_time}")

ae.compile(loss=loss, optimizer=opti)

history = ae.fit(dataset.noisy_train_ds, 
                validation_data=dataset.noisy_test_ds, 
                epochs=epochs, 
                callbacks=[logging_callback])

# save configs (e.g. hyperparameters) of your settings
hw_directory = str(Path(__file__).parents[0])
model_folder = 'my_model08'

dir = hw_directory + '/' + model_folder

ae.save(dir)

# plotting
#plt.plot(history.history["loss"])
#plt.plot(history.history["val_loss"])
#plt.legend(labels=["train_loss","val_loss"])
#plt.xlabel("Epoch")
#plt.ylabel("MSE(loss)")
#plt.savefig(f"Homework/08/plot/e={epochs},lr={lr}.png")
#plt.show()

# # save configs (e.g. hyperparameters) of your settings
# hw_directory = str(Path(__file__).parents[0])
# model_folder = 'my_model07'

# dir = hw_directory + '/' + model_folder

# ae.save(dir)
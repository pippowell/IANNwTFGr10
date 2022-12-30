import tensorflow as tf
import dataset 
import model
import matplotlib.pyplot as plt
import datetime as datetime
from pathlib import Path

# Initiate epochs and learning rate as global variables
epochs = 15
lr = 1e-2

mymodel = model.BasicCNN_LSTM()

loss = tf.keras.losses.MeanSquaredError()
opti = tf.keras.optimizers.Adam(learning_rate=lr)

mymodel.compile(loss=loss, 
                optimizer=opti, 
                metrics=['MAE'])  # for accuracy - instead of tf.keras.metrics.MeanAbsoluteError()
                      
# save logs with Tensorboard
EXPERIMENT_NAME = "CNN_LSTM"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logging_callback = tf.keras.callbacks.TensorBoard(log_dir=f".Homework/07/logs/{EXPERIMENT_NAME}/{current_time}")

history = mymodel.fit(dataset.train_ds, 
                    validation_data=dataset.val_ds, 
                    epochs=epochs,
                    batch_size=dataset.batch_size,
                    callbacks=[logging_callback])

# mymodel.load_weights(checkpoint_filepath)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.plot(history.history["MAE"])
plt.plot(history.history["val_MAE"])
plt.legend(labels=["train_loss","val_loss", "train_error(acc)", "val_error(acc)"])
plt.xlabel("Epoch")
plt.ylabel("MSE(loss), MAE(acc)")
plt.savefig(f"testing: e={epochs},lr={lr}.png")
plt.show()

# save configs (e.g. hyperparameters) of your settings
# checkpoint your model’s weights (or even the complete model)

hw_directory = str(Path(__file__).parents[0])
model_folder = 'my_model07'

dir = hw_directory + '/' + model_folder

mymodel.save(dir)


# try checkpoint when there's time:
# checkpoint_filepath = 'checkpoint.hdf5'
# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
#                                                                 save_weights_only=True,
#                                                                 monitor='val_accuracy',
                                                                # save_best_only=True)

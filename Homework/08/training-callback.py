import tensorflow as tf
import dataset
import model
import matplotlib.pyplot as plt
import datetime as datetime
from pathlib import Path
import keras

# Initiate epochs and learning rate as global variables
epochs = 2
lr = 1e-3

ae = model.autoencoder()

loss = tf.keras.losses.MeanSquaredError() # tf.keras.losses.BinaryCrossentropy()
opti = tf.keras.optimizers.Adam(learning_rate=lr)

# save logs with Tensorboard
EXPERIMENT_NAME = "CNN_LSTM"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logging_callback = tf.keras.callbacks.TensorBoard(log_dir=f".Homework/07/logs/{EXPERIMENT_NAME}/{current_time}")

ae.compile(loss=loss, optimizer=opti)

# custom Callback for visualization
class CustomCallback(keras.callbacks.Callback):
    def __init__(self, train=dataset.noisy_train_ds, validation=dataset.noisy_test_ds):
        super(CustomCallback, self).__init__()
        self.validation = validation
        self.train = train
    def on_test_batch_begin(self, batch=dataset.batchsize,logs=None):
        if (self.validation):
            x_valid, y_valid = self.validation[0], self.validation[1]
            y_val_pred = self.model.predict(x_valid)
            print(x_valid, y_val_pred)
        # if self.validation is None:
        #     print("a")
        # else:
        #     # Get the validation data
        #     x_val, y_val = self.validation_data
        #     print("b")
        # # Make predictions on the validation data
        # predictions = self.model.predict(x_val)
        #
        # # Loop over the predictions
        # for prediction, label in zip(predictions, y_val):
        #     # Display the prediction and the label
        #     plt.imshow(prediction[0])
        #     plt.title(label[0])
        #     plt.show()

    # def on_test_batch_end(self, batch, logs=None):
    #         print("12")



history = ae.fit(dataset.noisy_train_ds, 
                validation_data=dataset.noisy_test_ds, 
                epochs=epochs, 
                callbacks=[CustomCallback()]) #[logging_callback])

# plotting
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(labels=["train_loss","val_loss"])
plt.xlabel("Epoch")
plt.ylabel("MSE(loss)")
plt.savefig(f"e={epochs},lr={lr}.png")
plt.show()

# # save configs (e.g. hyperparameters) of your settings
# hw_directory = str(Path(__file__).parents[0])
# model_folder = 'my_model07'

# dir = hw_directory + '/' + model_folder

# ae.save(dir)
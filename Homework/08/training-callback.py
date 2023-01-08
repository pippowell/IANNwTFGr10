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

class CustomCallback(keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(CustomCallback, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs={}):
        batches = len(self.validation_data)  # 1. Get the amount of batches the generator provides
        # .. other code
        for batch in range(batches):  # 2. Iterate over the amount of batches
            xVal, label = next(iter(self.validation_data))  # 3. Retrieve a batch
            if batch % 50 == 0:
                predictions = self.model.predict(xVal) # This one is not correct (probably)
                zipped = zip(predictions, label)
                for i, (prediction, label) in enumerate(zipped):
                    # Display the prediction and the label
                    fig, ax = plt.subplots(nrows=1, ncols=2)
                    ax[0].imshow(prediction)
                    ax[1].imshow(label)
                    plt.show()



history = ae.fit(dataset.noisy_train_ds, 
                validation_data=dataset.noisy_test_ds, 
                epochs=epochs, 
                callbacks=[CustomCallback(dataset.noisy_test_ds)]) #[logging_callback])

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
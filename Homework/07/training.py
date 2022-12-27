import tensorflow as tf
import dataset 
import model
import matplotlib.pyplot as plt
import datetime as datetime


# # Initiate the logs and metrics
# config_name= "HW07"
# current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# train_log_path = f"logs/{config_name}/{current_time}/train"
# val_log_path = f"logs/{config_name}/{current_time}/val"

# # log writer for training metrics
# train_summary_writer = tf.summary.create_file_writer(train_log_path)

# # log writer for validation metrics
# val_summary_writer = tf.summary.create_file_writer(val_log_path)

# # Define arrays for saving values for later visualization
# train_forb_norm = []
# train_losses = []
# train_accuracies = []

# val_forb_norm = [] 
# val_losses = []
# val_accuracies = []

# mymodel.compile(run_eagerly=True) 

# Initiate epochs as global variables
epochs = 2 #15

mymodel = model.BasicCNN_LSTM()

loss = tf.keras.losses.MeanSquaredError()
opti = tf.keras.optimizers.Adam(learning_rate=1e-2)

mymodel.compile(loss=loss, 
                optimizer=opti, 
                metrics=['MAE']) # for accuracy - instead of tf.keras.metrics.MeanAbsoluteError()
                      

history = mymodel.fit(dataset.train_ds, 
                    validation_data=dataset.val_ds, 
                    epochs=epochs,
                    batch_size=dataset.batch_size,
                    callbacks=[logging_callback])

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.plot(history.history["MAE"])
plt.plot(history.history["val_MAE"])
plt.legend(labels=["train_loss","val_loss", "train_error(acc)", "val_error(acc)"])
plt.xlabel("Epoch")
plt.ylabel("MSE(loss), MAE(acc)")
plt.show()
plt.savefig("hw7")

# NEED TO MAKE LOAD_MODEL AND ALL THE SMALL STUFFS WORK

# EXPERIMENT_NAME = "CNN_LSTM"
# current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# logging_callback = tf.keras.callbacks.TensorBoard(log_dir=f"./logs/{EXPERIMENT_NAME}/{current_time}")

# mymodel.save("saved_model hw07")
# # load the model and resume training where we had to stop
# loaded_model = tf.keras.models.load_model("/07/saved_model hw07")
# history.history
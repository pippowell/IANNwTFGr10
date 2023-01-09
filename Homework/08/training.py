import tensorflow as tf
import dataset
import model
import matplotlib.pyplot as plt
import datetime as datetime
from pathlib import Path
from sklearn.manifold import TSNE

# Initiate epochs and learning rate as global variables
epochs = 2
lr = 1e-3

ae = model.autoencoder(vae=False)
vae = model.autoencoder(vae=True)

loss = tf.keras.losses.MeanSquaredError() # tf.keras.losses.BinaryCrossentropy()
opti = tf.keras.optimizers.Adam(learning_rate=lr)

# save logs with Tensorboard
EXPERIMENT_NAME = "CNN_LSTM"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logging_callback = tf.keras.callbacks.TensorBoard(log_dir=f".Homework/07/logs/{EXPERIMENT_NAME}/{current_time}")

ae.compile(loss=loss, optimizer=opti)

history = ae.fit(dataset.noisy_train_ds, 
                validation_data=dataset.noisy_test_ds, 
                epochs=epochs, 
                callbacks=[logging_callback])

tsne = TSNE(n_components=2)

dataset = dataset.noisy_test_ds[:1000]

ae_encoding = ae.encoder(dataset)

tsne_result = tsne.fit_transform(ae_encoding)

# Plot the result of our TSNE with the label color coded
tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y[:1000]})

plt.figure(figsize=(10,8))
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df)
plt.title("t-SNE with mnist_784")
plt.savefig("Homework/08/plot/t-sne_mnist784")
plt.show()






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
import tensorflow as tf
import dataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from keras.datasets import mnist

ae_reload = tf.keras.models.load_model('Homework/08/my_model:e=15,lr=1e-3')

tsne = TSNE(n_components=2)

# test_1000 = dataset.noisy_img_train[:1000]
# test_ds = dataset.noisy_img_train.with_format("tf")
# y = dataset.labels_test

(_, _), (test_X, test_Y) = mnist.load_data()
test_X = tf.expand_dims(test_X, axis=-1)
test_X = tf.cast(test_X, tf.float32)
test_X = test_X[:1000]

ae_encoder = ae_reload.encoder

output_of_encoder = ae_encoder(test_X, False)

tsne_result = tsne.fit_transform(output_of_encoder)

# Plot the result of our TSNE with the label color coded
tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': test_Y[:1000]})

plt.figure(figsize=(10,8))
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df)
plt.title("t-SNE with 1000 testds")
plt.savefig("Homework/08/plot/tsne_1000testds")
plt.show()
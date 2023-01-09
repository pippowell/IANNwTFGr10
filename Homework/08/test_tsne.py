import tensorflow as tf
import dataset
import datasetwlabel
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns



ae_reload = tf.keras.models.load_model('Homework/08/my_ae')

tsne = TSNE(n_components=2)

test_1000 = dataset.noisy_test_ds[:1000]
y = datasetwlabel.testlabels

ae_encoding = ae_reload.encoder(test_1000)

tsne_result = tsne.fit_transform(ae_encoding)

# Plot the result of our TSNE with the label color coded
tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y[:1000]})

plt.figure(figsize=(10,8))
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df)
plt.title("t-SNE with mnist_784")
plt.savefig("Homework/08/plot/t-sne_mnist784")
plt.show()
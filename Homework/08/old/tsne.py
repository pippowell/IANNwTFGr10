from sklearn import datasets
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

# WE NEED TO DO IT WITH TEST DATA
X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')

# first 1000 images of the data set
X = X[:1000] # from X.shape=(70000, 784) to X.shape=(1000, 784)

# Fit and transform with a TSNE
n_components = 2
tsne = TSNE(n_components)

# Project the data in 2D
tsne_result = tsne.fit_transform(X) # tsne_result.shape=(1000, 2): Two dimensions for each of our images
 
# Plot the result of our TSNE with the label color coded
tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y[:1000]})

plt.figure(figsize=(10,8))
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df)
plt.title("t-SNE with mnist_784")
plt.savefig("Homework/08/plot/t-sne_mnist784")
plt.show()
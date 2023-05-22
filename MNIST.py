import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from keras.datasets import mnist

# mnist = fetch_openml("mnist_784", version=1, parser='auto')

# x_train, y_train = mnist.data / 255., mnist.target
# x_test, y_test = mnist.data / 255., mnist.target
# x_train = x_train.values.reshape((len(x_train), -1))
# x_test = x_test.values.reshape((len(x_test), -1))

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape)
print(x_test.shape)

# pca = PCA(n_components=30).fit(x_train)
# reduced_X = pca.transform(x_train)
#
# model = KMeans(init="k-means++", n_clusters=10, random_state=0, n_init=10)
# model.fit(x_train)
#
# y_pred = model.labels_

def viz_img(y_pred):
    n = 10
    fig = plt.figure(1)
    box_index = 1
    for cluster in range(10):
        result = np.where(y_pred == cluster)
        for i in np.random.choice(result[0].tolist(), n, replace=False):
            ax = fig.add_subplot(n, n, box_index)
            plt.imshow(x_train[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            box_index += 1
    plt.show()

# viz_img(y_pred)

#============================================================================

tsne = TSNE(learning_rate=300)
transformed = tsne.fit_transform(x_train)

model = DBSCAN(eps=2.4, min_samples=100)
predict = model.fit(transformed)
# pd.Series(predict.labels_).value_counts()

y_pred = predict.labels_

dataset = pd.DataFrame({'Column1':transformed[:,0],'Column2':transformed[:,1]})
dataset['cluster_num'] = pd.Series(predict.labels_)

# Cluster Viz 1
plt.rcParams['figure.figsize'] = [5, 4]
plt.scatter(dataset[dataset['cluster_num'] == 0]['Column1'],
            dataset[dataset['cluster_num'] == 0]['Column2'],
            s = 50, c = 'red', label = 'Customer Group 1')
plt.scatter(dataset[dataset['cluster_num'] == 1]['Column1'],
            dataset[dataset['cluster_num'] == 1]['Column2'],
            s = 50, c = 'orange', label = 'Customer Group 2')
plt.scatter(dataset[dataset['cluster_num'] == 2]['Column1'],
            dataset[dataset['cluster_num'] == 2]['Column2'],
            s = 50, c = 'yellow', label = 'Customer Group 3')
plt.scatter(dataset[dataset['cluster_num'] == 3]['Column1'],
            dataset[dataset['cluster_num'] == 3]['Column2'],
            s = 50, c = 'green', label = 'Customer Group 4')
plt.scatter(dataset[dataset['cluster_num'] == 4]['Column1'],
            dataset[dataset['cluster_num'] == 4]['Column2'],
            s = 50, c = 'blue', label = 'Customer Group 5')
plt.scatter(dataset[dataset['cluster_num'] == 5]['Column1'],
            dataset[dataset['cluster_num'] == 5]['Column2'],
            s = 50, c = 'darkblue', label = 'Customer Group 6')
plt.scatter(dataset[dataset['cluster_num'] == 6]['Column1'],
            dataset[dataset['cluster_num'] == 6]['Column2'],
            s = 50, c = 'purple', label = 'Customer Group 7')
plt.scatter(dataset[dataset['cluster_num'] == 7]['Column1'],
            dataset[dataset['cluster_num'] == 7]['Column2'],
            s = 50, c = 'gray', label = 'Customer Group 8')
plt.scatter(dataset[dataset['cluster_num'] == 8]['Column1'],
            dataset[dataset['cluster_num'] == 8]['Column2'],
            s = 50, c = 'black', label = 'Customer Group 9')
plt.scatter(dataset[dataset['cluster_num'] == 9]['Column1'],
            dataset[dataset['cluster_num'] == 9]['Column2'],
            s = 50, c = 'magenta', label = 'Customer Group 10')
plt.title('Type Of Keyword')
plt.show()

viz_img(y_pred)
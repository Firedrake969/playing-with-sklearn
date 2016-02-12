import numpy as np

from sklearn import datasets, cluster

iris = datasets.load_iris()
n_samples = len(iris.data)
data = iris.data.reshape((n_samples, -1))

model = cluster.DBSCAN()
# model = cluster.MeanShift()
model.fit(data)

# gives number of predicted clusters
print 'Predicted:  ', len(np.unique(model.labels_))
print 'Actual:     ', len(iris.target_names)
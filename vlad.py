import numpy as np
import numpy.linalg as LA
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import six
from sklearn.metrics.pairwise import pairwise_distances

class Vlad:
    def __init__(self, num_clusters = 8):
        self.num_clusters = num_clusters
        return

    def my_vlad(self, local_descriptors, centroids):
        #print(centroids.shape, local_descriptors.shape)
        V = np.zeros([centroids.shape[0],local_descriptors.shape[1]])
        print(V.shape, centroids.shape, local_descriptors.shape)
        distances = pairwise_distances(local_descriptors, centroids, metric='euclidean')
        clusters = np.argmin(distances,axis=1)
        for iter, center in enumerate(centroids):
            points_belonging_to_cluster = local_descriptors[clusters == iter]
            V[iter] = np.sum(points_belonging_to_cluster - center, axis=0)
        V = V.reshape(1, V.shape[0]*V.shape[1])
        #print ('Vlad shape is ', V.shape)
        return V/LA.norm(V)

    def get_params(self, deep=True):
        return dict(num_clusters = self.num_clusters)

    def set_params(self, **params):
        if not params:
            return self
        for key, value in six.iteritems(params):
            split = key.split('__', 1)
            if len(split) > 1:
                print("length is greter than one ", split, value)
            else:
                print("length is one ", split, value)
                setattr(self, key, value)

    def fit(self, X, y=None):
        print("in fit method", X.shape, y.shape, self.num_clusters)
        tmp = X.swapaxes(1,2)
        tmp = tmp.reshape(tmp.shape[0]*tmp.shape[1], tmp.shape[2])
        kmeans = MiniBatchKMeans(init='k-means++', n_clusters=self.num_clusters, batch_size=1000)
        kmeans.fit(tmp)
        self.centers = kmeans.cluster_centers_
        self.labels = kmeans.labels_
        print("shape of centers is ",self.centers.shape)
        return self

    def transform(self, X):
        print("in transform method", X.shape, self.num_clusters)
        X = X.swapaxes(1,2)
        tot_range = X.shape[0]
        print("X.shape is ", X.shape)
        out = np.empty((tot_range, self.centers.shape[0]*X.shape[2]))
        for i in range(tot_range):
            out[i] = self.my_vlad(X[i], self.centers)

        out = np.insert(out, 0, 1, axis=1)
        print(out.shape)
        return out


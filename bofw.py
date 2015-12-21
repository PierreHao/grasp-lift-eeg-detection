import numpy as np
import numpy.linalg as LA
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import six
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler

class Bofw:
    def __init__(self, num_clusters = 8):
        self.num_clusters = num_clusters
        return

    def my_bofw(self, loc_desc, centroids, clusters):
        B = np.zeros(centroids.shape[0])
        local_descriptors = self.scaler.transform(loc_desc)
        for iter, center in enumerate(centroids):
            points_belonging_to_cluster = local_descriptors[clusters == iter]
            B[iter] = points_belonging_to_cluster.shape[0]
        return B

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
        tmp = X.swapaxes(1,2)
        tmp = tmp.reshape(tmp.shape[0]*tmp.shape[1], tmp.shape[2])

        self.scaler = StandardScaler()
        self.scaler.fit(tmp)
        tmp = self.scaler.transform(tmp)
        kmeans = MiniBatchKMeans(init='k-means++', n_clusters=self.num_clusters, batch_size=1000)
        kmeans.fit(tmp)
        self.centers = kmeans.cluster_centers_
        self.clusters = kmeans.labels_
        print("shape of centers is ",self.centers.shape)
        return self

    def transform(self, X):
        print("in transform method", X.shape, self.num_clusters)
        X = X.swapaxes(1,2)
        tot_range = X.shape[0]
        print("X.shape is ", X.shape)
        out = np.empty((tot_range, self.centers.shape[0]))
        print("starting for loop")
        print tot_range
	start_ind = 0
        for i in range(tot_range):
            out[i] = self.my_bofw(X[i], self.centers, self.clusters[start_ind:start_ind + 500])
            start_ind = start_ind + 500
        out = np.insert(out, 0, 1, axis=1)
        return out

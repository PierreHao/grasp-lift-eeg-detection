import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#Load and scale data. just for one file. Write the for loop for all file 
X = np.load("../data/processed/subj1_train_data.npy")
events,dim,vectors = X.shape
scaler = StandardScaler()

#PCA
pca = PCA(n_components=10)

#MiniBatch Kmeans:
minibatch = MiniBatchKMeans(n_clusters=10,max_iter=100,batch_size=33)

#Pipeline:
galPPLine = Pipeline(steps=[
        ('pca', pca),
        ('minibatch', minibatch)
        ])

for i in range(events):
        scaler.fit(X[i])
        X_scaled = scaler.transform(X[i])
        fittedData = galPPLine.set_params(pca__n_components=10,
                       minibatch__n_clusters=3,
                       minibatch__batch_size=10).fit(X_scaled)
        break

testVector = X[0].T[0]
print testVector.reshape(-1, 1).shape
print fittedData.predict(testVector.reshape(-1, 1))

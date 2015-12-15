import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.externals import six
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from vlad import Vlad

DATA_DIR = "data/processed/"

subjects = range(1, 3)

X =  np.concatenate([np.load("{0}/subj{1}_train_data.npy".format(DATA_DIR, subject)) for subject in subjects])
y =  np.concatenate([np.load("{0}/subj{1}_train_labels.npy".format(DATA_DIR, subject)) for subject in subjects])

X_test =  np.concatenate([np.load("{0}/subj{1}_val_data.npy".format(DATA_DIR, subject)) for subject in subjects])
y_test =  np.concatenate([np.load("{0}/subj{1}_val_labels.npy".format(DATA_DIR, subject)) for subject in subjects])

y = y[:, 2]
y_test = y_test[:,2]

print(X.shape, y.shape)
print(X_test.shape, y_test.shape)

clf = svm.SVC(kernel='linear')
myVlad = Vlad()
pca = PCA(n_components=0.9)
scaler = StandardScaler()

vlad_pipeline = Pipeline([('myown', myVlad), ('vlad_pca', pca), ('vlad_scaling', scaler), ('svm', clf)])

#num_clusters = [2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9 ]
num_clusters = [2**5]
estimator = GridSearchCV(vlad_pipeline, dict(myown__num_clusters=num_clusters))
estimator.fit(X,y)
estimator.predict(X_test)

score = estimator.score(X_test, y_test)
predictions = estimator.predict(X_test)

print(score)

y_binary = label_binarize(y_test,classes=[1,2,3,4,5,6])
predictions_binary=label_binarize(predictions,classes=[1,2,3,4,5,6])

aucTotal = 0
for i in range(0,6):
	singleAuc=roc_auc_score(y_binary[:,i],predictions_binary[:,i])
	aucTotal+=singleAuc
	print("for label",i,"auc=",singleAuc)

print("ACU score ", aucTotal/6)

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

import sys
import csv

DATA_DIR = "data/processed/"
N_COMPONENT = 3

subjects = range(1, 13)

X =  np.concatenate([np.load("{0}/{1}/subj{2}_train_data.npy".format(DATA_DIR, N_COMPONENT, subject)) for subject in subjects])
y =  np.concatenate([np.load("{0}/{1}/subj{2}_train_labels.npy".format(DATA_DIR, N_COMPONENT, subject)) for subject in subjects])

X_test =  np.concatenate([np.load("{0}/{1}/subj{2}_val_data.npy".format(DATA_DIR, N_COMPONENT, subject)) for subject in subjects])
y_test =  np.concatenate([np.load("{0}/{1}/subj{2}_val_labels.npy".format(DATA_DIR, N_COMPONENT, subject)) for subject in subjects])

y = y[:, 2]
y_test = y_test[:,2]

print(X.shape, y.shape)
print(X_test.shape, y_test.shape)

clf = svm.SVC(kernel='rbf',C=1)
myVlad = Vlad()
pca = PCA(n_components=0.9)
scaler = StandardScaler()

vlad_pipeline = Pipeline([('myown', myVlad), ('vlad_pca', pca), ('vlad_scaling', scaler), ('svm', clf)])

num_clusters = [2**8, 2**9, 2**10, 2**11]
#cGrid= [2**8,2**9,2**10,2**11]
#cGrid=[2**3]
cGrid=[2**-4,2**-3, 2**-2, 2**-1, 2**0, 2**1, 2**2, 2**3,2**4]

estimator = GridSearchCV(vlad_pipeline, dict(myown__num_clusters=num_clusters,svm__C=cGrid), n_jobs = 12,verbose=100)
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

fileName = "AUC_"+str(N_COMPONENT)+"components.csv"
with open(fileName, "w") as myfile:
    myfile.write("best estimator:"+str(estimator.best_score_)+"\n")
    writer = csv.writer(myfile, delimiter = ",")
    paramKeys = list(estimator.grid_scores_[0].parameters.keys())

    writer.writerow(['mean']+ paramKeys)
    
    for i in estimator.grid_scores_:
        output = list()
        output.append(i.mean_validation_score)

        for k in paramKeys:
            output.append(i.parameters.get(k))

        writer.writerow(output)

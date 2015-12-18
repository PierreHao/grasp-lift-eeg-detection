import datetime
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
from bofw import Bofw
import sys


print("Start time is ", datetime.datetime.now())

DATA_DIR = "/home/mc3784/grasp-lift-eeg-detection/data/processed"
N_COMPONENT = 2

subjects = range(1, 13)

X =  np.concatenate([np.load("{0}/{1}/subj{2}_train_data.npy".format(DATA_DIR, N_COMPONENT, subject)) for subject in subjects])
y =  np.concatenate([np.load("{0}/{1}/subj{2}_train_labels.npy".format(DATA_DIR, N_COMPONENT, subject)) for subject in subjects])

X_test =  np.concatenate([np.load("{0}/{1}/subj{2}_val_data.npy".format(DATA_DIR, N_COMPONENT, subject)) for subject in subjects])
y_test =  np.concatenate([np.load("{0}/{1}/subj{2}_val_labels.npy".format(DATA_DIR, N_COMPONENT, subject)) for subject in subjects])

y = y[:, 2]
y_test = y_test[:,2]

print(X.shape, y.shape)
print(X_test.shape, y_test.shape)

clf = svm.SVC(kernel='linear',C=1)
myBofw = Bofw()
pca = PCA(n_components=0.9)
scaler = StandardScaler()

bofw_pipeline = Pipeline([('myown', myBofw), ('bofw_pca', pca), ('bofw_scaling', scaler), ('svm', clf)])

#num_clusters = [2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9 ]
num_clusters = [2**8,2**9, 2**10, 2**11]
cGrid=[2**-4, 2**-3, 2**-2, 2**-1, 2**0, 2**1, 2**2, 2**3, 2**4]
estimator = GridSearchCV(bofw_pipeline, dict(myown__num_clusters=num_clusters,svm__C=cGrid), n_jobs =12 )
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

fileName = "AUC_"+N_COMPONENT+"components.csv"
with open(fileName, "a") as myfile:
    myfile.write("best estimator:"+str(estimator.best_score_))  	
    myfile.write("svm__C, myown__num_clusters,mean_validation_score,cv_validation_scores\n")
    for i in estimator.grid_scores_:
	#fileLine= i, "splitValues: ",estimator.grid_scores_[0].cv_validation_scores
	#myfile.write(fileLine)
	#myfile.write("/n")
	#print fileLine
	myfile.write(str(i.parameters.get("svm__C")))
	myfile.write(", ")
	myfile.write(str(i.parameters.get("myown__num_clusters")))
        myfile.write(", ")
        myfile.write(str(i.mean_validation_score))
        myfile.write(", ")
        myfile.write(str(i.cv_validation_scores)[1:-1])
	myfile.write("\n")

print("End time is ", datetime.datetime.now())

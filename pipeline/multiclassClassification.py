import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


#Load and scale data. just for one file. Write the for loop for all file 
X = np.load("../data/processed/subj1_train_data.npy")
label = np.load("../data/processed/subj1_train_labels.npy")
y = label[:,2]

X_test = np.load("../data/processed/subj1_val_data.npy")
label = np.load("../data/processed/subj1_val_labels.npy")
y_test = label[:,2]



# Eliminate one of the dimension in order to user SVM: 
X = np.mean(X,axis=2)
X_test = np.mean(X_test,axis=2)
#To split the data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33,random_state=42)


#Use one of the two following version (commenting the other)
classifier = LinearSVC(random_state=0)
predictions = classifier.fit(X, y).predict(X_test)

y_binary = label_binarize(y_test,classes=[1,2,3,4,5,6])
predictions_binary=label_binarize(predictions,classes=[1,2,3,4,5,6])

aucTotal = 0
for i in range(0,6):
	singleAuc=roc_auc_score(y_binary[:,i],predictions_binary[:,i])
	aucTotal+=singleAuc
	print "for label",i,"auc=",singleAuc	

print y [1:50]
print predictions[1:50] 
print aucTotal/6


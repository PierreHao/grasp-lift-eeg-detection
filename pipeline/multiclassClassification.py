import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


#Load and scale data. just for one file. Write the for loop for all file 
X = np.load("../data/processed/subj1_train_data.npy")
events,dim,vectors = X.shape
label = np.load("../data/processed/subj1_train_labels.npy")
y = label[:,2]
# Eliminate one of the dimension in order to user SVM: 
X = np.mean(X,axis=2)
#To split the data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33,random_state=42)


#Use one of the two following version (commenting the other)
#classifier = LinearSVC(random_state=0)
#predictions = classifier.fit(X, y).predict(X)
y = label_binarize(y,classes=[1,2,3,4,5,6])
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=0))

predictions = classifier.fit(X, y).predict(X)


y_score = classifier.fit(X, y).decision_function(X)

print(y [1:50])
print(predictions[1:50])
print( y_score)

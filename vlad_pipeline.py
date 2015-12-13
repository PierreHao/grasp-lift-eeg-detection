import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn import svm


class Vlad:
    def __init__(self):
        return

    def fit(self, X, y=None):
        print("in fit method", X.shape, y.shape)
        return self

    def transform(self, X):
        print("in transform method", X.shape)
        return X.mean(axis=2)


DATA_DIR = "data/processed/"

subject  = 1

train_data = "subj9_train_data.npy"

X = np.load("{0}/subj{1}_train_data.npy".format(DATA_DIR, subject))
y = np.load("{0}/subj{1}_train_labels.npy".format(DATA_DIR, subject))
y = y[:, 2]

print(X.shape)
print(y.shape)


clf = svm.SVC(kernel='linear')
myVlad = Vlad()

vlad_pipeline = Pipeline([('myown', myVlad), ('svm', clf)])
vlad_pipeline.fit(X,y)
#vlad_pipeline.predict(X)

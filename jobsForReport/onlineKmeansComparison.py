import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from memory_profiler import memory_usage
import timeit
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import MiniBatchKMeans
import math
import os
import sys
print "Starting script..."

directory = "plots_"+sys.argv[1]
if not os.path.exists(directory):
    os.makedirs(directory)

def getPermutation(totalRange,numberElements):
    random_seed = 10312003
    rng = np.random.RandomState(random_seed)
    permutation = rng.permutation(totalRange)
    return permutation[:numberElements]

def onlineKmeans(X,k=3,b=30,maxiter=1000):
    centroids = generateStartingCentroid(X,k)
    pointsPerClusters = np.zeros([k,1])
    minInertia=0
    notImproveCount=0
    for i in range(maxiter):
        M=X[getPermutation(len(X),b)]
        distances = pairwise_distances(M, centroids, metric='euclidean')
        nearestCenters = np.argmin(distances, axis=1)
        inertia = np.sum(np.amin(distances, axis=1))/b
        if (minInertia ==0 or inertia < minInertia):
            minInertia = inertia
            notImproveCount=0
        elif minInertia !=0 and inertia >= minInertia:
            notImproveCount+=1
        if (notImproveCount>10):
            print "Last iteration : ", i, ", inertia: ",inertia, ", minInertia:",minInertia
            break
        for iter, x in enumerate(M):
            centerIndex = nearestCenters[iter]
            pointsPerClusters[centerIndex] = pointsPerClusters[centerIndex] + 1
            eta = 1/pointsPerClusters[centerIndex]
            centroids[centerIndex] = (1 - eta)*centroids[centerIndex] + eta * x

    return centroids
def generateStartingCentroid(X,maxClusternumbers):
    randState = 10312003
    rnd = np.random.RandomState(randState)
    nPoint,dimension = X.shape
    centroids=np.zeros([maxClusternumbers,dimension])
    centroids[0] = X[rnd.permutation(len(X))[0]]
    for i in (range(1,maxClusternumbers)):
        distances = pairwise_distances(X, centroids[:i], metric='euclidean')
        d2weighting=np.power(np.min(distances,axis=1),2)
        d2weighting = d2weighting/np.sum(d2weighting)
        allIndex = range(len(X))
        index = np.random.choice(allIndex, p=d2weighting)
        centroids[i]=X[index]
    return centroids



def profile_memory_and_time(function, *args, **kwargs):
    start_time = timeit.default_timer()
    memory, return_val = memory_usage((function, (args), kwargs), max_usage=True, retval=True)
    elapsed = timeit.default_timer() - start_time
    return memory[0], elapsed,return_val


nClusters = 1024#Number of clusters
batchSize = nClusters*3
nMaxIter = 1000
runInfo="ncluster= "+str(nClusters)+" batchSize= "+str(batchSize)
dim=32

minibatch=MiniBatchKMeans(n_clusters=nClusters,max_iter=nMaxIter,batch_size=batchSize)
inputSizesToGenerate = np.array([2**12,2**14,2**16,2**18,2**20,2**22])
#[[2**8,32],[2**10,32],[2**12,32]] 
#[[2**8, 32],[2**10, 32],[2**12, 32],[2**14, 32],[2**16, 32],[2**18, 32],[2**20, 32],[2**22, 32],[2**24, 32]]
            #,[2**26, 32],[2**28, 32],[2**30, 32],[2**32, 32]]

scaler = StandardScaler()
runResults = np.zeros([len(inputSizesToGenerate),5])

for nEnum,num_samples in enumerate(inputSizesToGenerate):
    print "Running for {0} samples of dimension {1}".format(num_samples, dim)
    X,y = make_blobs(n_samples=num_samples, n_features=dim, centers=10)
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    ourMemory,ourTime,ourCentroids= profile_memory_and_time(onlineKmeans, X, k=nClusters, b=batchSize, maxiter=nMaxIter)
    memory, time, rval = profile_memory_and_time(minibatch.fit,X)
    runResults[nEnum] = math.log(num_samples,2),ourMemory,ourTime,memory, time
    lnSample = math.log(num_samples,2)

#Plot results:    
plt.scatter(runResults[:,0], runResults[:,1],color="green",label="online Keameans")
plt.scatter(runResults[:,0], runResults[:,3],color="red",label="sklearn Kmeans")
plt.xlabel('Number of sample')
plt.ylabel('Memory usage')
title="MEMORY for "+str(runInfo)
plt.title(title)
plt.legend(loc=2)
plt.show()

plt.savefig('{}/memory.png'.format(directory))
plt.clf()
plt.scatter(runResults[:,0], runResults[:,2],color="green",label="online Keameans")
plt.scatter(runResults[:,0], runResults[:,4],color="red",label="sklearn Kmeans")
plt.xlabel('Number of sample')
plt.ylabel('Time')
title="TIME for "+str(runInfo)
plt.title(title)
plt.legend(loc=2)
plt.show()

plt.savefig('{}/time.png'.format(directory))
print "Script ended. Results:"
print "numSamples: ",runResults[:,0]
print "ourMemories: ",runResults[:,1]
print "kmeansMemory: ",runResults[:,3]
print "ourTimes: ", runResults[:,2]
print "kmeansTimes: ",runResults[:,4]
print runResults

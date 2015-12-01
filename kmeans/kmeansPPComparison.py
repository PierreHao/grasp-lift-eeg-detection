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
print "Starting script..."


from timeit import Timer
from functools import partial

def getPermutation(totalRange,numberElements):
    random_seed = 10312003
    rng = np.random.RandomState(random_seed)
    permutation = rng.permutation(totalRange)
    return permutation[:numberElements]



def mykmeans_plus_plus(X, max_clusters = 8, max_iterations=1000):
    randState = 10312003
    rnd = np.random.RandomState(randState)
    centroids=generateStartingCentroid(X,max_clusters)

    minInertia = 0
    inertia = 0
    for nIter in range(0, max_iterations):
        distances = pairwise_distances(X, centroids, metric='euclidean')
        clusters = np.argmin(distances,axis=1)
        min_distances = np.amin(distances, axis=1)
        sum_distortion = min_distances.sum()
        inertia = np.sum(np.amin(distances, axis=1))  
        if (minInertia ==0 or inertia < minInertia):
            minInertia = inertia
            notImproveCount=0
        elif minInertia !=0 and inertia >= minInertia:
            notImproveCount+=1
        if (notImproveCount>10):
            print "iteration: ", nIter, " inertia: ",inertia, " minInertia: ",minInertia   
            break
        data = np.concatenate([X, clusters[:,np.newaxis]], axis=1)
        for cRange in range(0,max_clusters):
            allpoints = data[np.where(data[:,(data.shape[1] - 1)] == cRange)][:,range(0, data.shape[1] -1)]
            centroids[cRange] = np.sum(allpoints, axis=0)/allpoints.shape[0]
    return centroids


def generateStartingCentroid(X,maxClusternumbers):
    nPoint,dimension = X.shape
    centroids=np.zeros([maxClusternumbers,dimension])
    getPermutation
    randState = 10312003
    rnd = np.random.RandomState(randState)
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
    #print "args is ",args
    start_time = timeit.default_timer()
    memory, return_val = memory_usage((function, (args), kwargs), max_usage=True, retval=True)
    elapsed = timeit.default_timer() - start_time
    return memory[0], elapsed,return_val

from sklearn.cluster import KMeans
kmeans_plus_plus=KMeans(n_clusters=10, init='k-means++', n_init=10, max_iter=100)

inputSizesToGenerate = [[2**8, 32],[2**10, 32],[2**12, 32],[2**14, 32],[2**16, 32],[2**18, 32],[2**20, 32]]
			#,[2**22, 32],[2**24, 32],[2**26, 32],[2**28, 32],[2**30, 32],[2**32, 32]];

scaler = StandardScaler()
plt.ion()
f1 = plt.figure()
ax1 = f1.add_subplot(111)
ourTimes=[]
times=[]
ourMemories=[]
memoris=[]
numSamples=[]

for num_samples, num_dimension in inputSizesToGenerate:
    print "Running for {0} samples of dimension {1}".format(num_samples, num_dimension)
    X,y = make_blobs(n_samples=num_samples, n_features=num_dimension, centers=6)
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    ourMemory,ourTime,ourCentroids= profile_memory_and_time(mykmeans_plus_plus, X, max_clusters = 10, max_iterations=1000)
    ourTimes.append(ourTime)
    ourMemories.append(ourMemory)
    memory, time, rval = profile_memory_and_time(kmeans_plus_plus.fit,X)
    times.append(time)
    memoris.append(memory)
    numSamples.append(num_samples)
plt.scatter(numSamples, ourMemories,color="green",label=" Our Keameans++")
plt.scatter(numSamples, memoris,color="red",label="sklearn Kmeans++")
plt.title("MEMORY")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
plt.savefig('memory.png')
plt.clf()
plt.scatter(numSamples, ourTimes,color="green",label="Our Keameans++")
plt.scatter(numSamples, times,color="red",label="sklearn Kmeans++")
plt.title("TIME")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
plt.savefig('time.png')
print "Script ended. Results:"
print "numSamples: ",numSamples
print "ourMemories: ",ourMemories
print "kmeansMemory: ",memoris
print "ourTimes: ", ourTimes
print "kmeansTimes: ",times
    

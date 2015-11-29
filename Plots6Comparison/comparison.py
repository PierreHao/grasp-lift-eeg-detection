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
import datetime
print "Starting script..."
print"comparison our online with our plusplus for nsamples, ncluster: [[2**8, 2**2],[2**10, 2**4],[2**12, 2**6],[2**14, 2**8],[2**16, 2**10],[2**18, 2**12],[2**20, 2**14]]"
def getPermutation(totalRange,numberElements):
    random_seed = 10312003
    rng = np.random.RandomState(random_seed)
    permutation = rng.permutation(totalRange)
    return permutation[:numberElements]

def mykmeans_plus_plus(X, max_clusters = 8, maxiter=10000):
    randState = 10312003
    rnd = np.random.RandomState(randState)
    centroids=generateStartingCentroid(X,max_clusters)

    minInertia = 0
    inertia = 0
    for nIter in range(0, maxiter):
        distances = pairwise_distances(X, centroids, metric='euclidean')
        clusters = np.argmin(distances,axis=1)
        inertia = np.sum(np.amin(distances, axis=1))  
        if (minInertia ==0 or inertia < minInertia):
            minInertia = inertia
            notImproveCount=0
        elif minInertia !=0 and inertia >= minInertia:
            notImproveCount+=1
        if (notImproveCount>10):
            print datetime.datetime.now().time().isoformat(),"(plusplus) Last iteration: ", nIter, " inertia: ",inertia, " minInertia: ",minInertia   
            break
        data = np.concatenate([X, clusters[:,np.newaxis]], axis=1)
        for cRange in range(0,max_clusters):
            allpoints = data[np.where(data[:,(data.shape[1] - 1)] == cRange)][:,range(0, data.shape[1] -1)]
            centroids[cRange] = np.sum(allpoints, axis=0)/allpoints.shape[0]
    return centroids

def onlineKmeans(X,max_clusters=3,b=30,maxiter=1000):
    centroids = generateStartingCentroid(X,max_clusters)
    pointsPerClusters = np.zeros([max_clusters,1])
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
            print datetime.datetime.now().time().isoformat(),"(Online) Last iteration : ", i, ", inertia: ",inertia, ", minInertia:",minInertia
            break
        for iter, x in enumerate(M):
            centerIndex = nearestCenters[iter]
            pointsPerClusters[centerIndex] = pointsPerClusters[centerIndex] + 1
            eta = 1/pointsPerClusters[centerIndex]
            centroids[centerIndex] = (1 - eta)*centroids[centerIndex] + eta * x
        oldInertia = inertia

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

numberOfSamples=2**22
inputSizesToGenerate = [[2**8, 2**2],[2**10, 2**4],[2**12, 2**6],[2**14, 2**8],[2**16, 2**10],[2**18, 2**12],[2**20, 2**14]]
            #,[2**16, 32],[2**18, 32],[2**20, 32],[2**22, 32],[2**24, 32]]
            #,[2**26, 32],[2**28, 32],[2**30, 32],[2**32, 32]]

scaler = StandardScaler()
plt.ion()
f1 = plt.figure()
ax1 = f1.add_subplot(111)
onlineTimesList=[]
onlineMemoryList=[]
ppTimesList=[]
ppMemoriesList=[]
numSamples=[]
    
def savePlot(x,onlineY,ppY, title):
    plt.scatter(x, onlineY,color="green",label="online Keameans")
    plt.scatter(x, ppY,color="red",label="pp Kmeans")
    plt.title(title+" green=online, red=plusplus")
    #plt.legend(loc='upper right')
    plt.show()
    plt.savefig(title+'.png')
    plt.clf()

for num_samples, num_cluster in inputSizesToGenerate:
    print datetime.datetime.now().time().isoformat(),"Running for {0} samples of dimension {1} and nCluster {2}".format(num_samples, 32,num_cluster)
    X,y = make_blobs(n_samples=num_samples, n_features=32, centers=10)
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    onlineMemory,onlineTime,onlineCentroids = profile_memory_and_time(onlineKmeans, X, max_clusters=10, b=33, maxiter=1000)
    onlineTimesList.append(onlineTime)
    onlineMemoryList.append(onlineMemory)
    
    ppMemory,ppTime,ppCentroids = profile_memory_and_time(mykmeans_plus_plus, X, max_clusters=10, maxiter=1000)
    ppTimesList.append(ppTime)
    ppMemoriesList.append(ppMemory)
    
    numSamples.append(np.log2(num_cluster))
    numSamples=[2,4,6,8,10,12,14]

savePlot(numSamples,onlineMemoryList,ppMemoriesList,"memory") 
plt.clf()
savePlot(numSamples,onlineTimesList,ppTimesList,"time") 

print "Script ended. Results:"
print "numSamples: ",numSamples
print "onlineMemory: ",onlineMemory
print "onlineTimesList: ",onlineTimesList
print "ppMemoriesList: ", ppMemoriesList
print "ppTimesList: ",ppTimesList    


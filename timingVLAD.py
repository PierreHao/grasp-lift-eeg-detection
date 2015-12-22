import numpy as np
import numpy.linalg as LA
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import six
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import timeit
from memory_profiler import memory_usage
from vlad import Vlad
import math

def profile_memory_and_time(function, *args, **kwargs):
    start_time = timeit.default_timer()
    memory, return_val = memory_usage((function, (args), kwargs), max_usage=True, retval=True)
    elapsed = timeit.default_timer() - start_time
    return memory[0], elapsed,return_val

inputSizesToGenerate =[[2**15,2**10],[2**17,2**10],[2**19,2**10]]
scaler = StandardScaler()
times=[]
memoris=[]
numSamples=[]
num_dimension=32

N_COMPONENT = 2
DATA_DIR = "data/processed"
subjects = range(1, 5)

X =  np.concatenate([np.load("{0}/{1}/subj{2}_train_data.npy".format(DATA_DIR, N_COMPONENT, subject)) for subject in subjects])
print X.shape



for num_samples, num_cluster in inputSizesToGenerate:
    X=X[:num_samples]
    myVlad = Vlad(num_cluster)
    print "Running for {0} samples of dimension {1}".format(num_samples, num_dimension)
    memory, time, rval = profile_memory_and_time(myVlad.fit,X)
    times.append(time)
    memoris.append(memory)
    numSamples.append(math.log(num_samples,10))
    
plt.scatter(numSamples, memoris,color="red")
plt.title("VLAD: memory usage for k = 2**2")
plt.xlabel("number of samples")
plt.ylabel("memory")
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('memoryVaringSamples.png')
plt.clf()
plt.scatter(numSamples, times,color="red")
plt.title("VLAD: time usage for k = 2**2")
plt.xlabel("number of samples")
plt.ylabel("time")
plt.savefig('timeVaringSamples.png')
print "Script ended. Results:"
print "numSamples: ",numSamples
print "vladMemory: ",memoris
print "vladTimes: ",times



plt.clf()

inputSizesToGenerate =[[2**17,2**9],[2**17,2**10]]
scaler = StandardScaler()
times=[]
memoris=[]
numSamples=[]
X=X[:num_samples]

for num_samples, num_cluster in inputSizesToGenerate:
    myVlad = Vlad(num_cluster)
    print "Running for {0} samples of dimension {1}".format(num_samples, num_dimension)
    memory, time, rval = profile_memory_and_time(myVlad.fit,X)
    times.append(time)
    memoris.append(memory)
    numSamples.append(math.log(num_cluster,10))

plt.scatter(numSamples, memoris,color="red")
plt.title("VLAD: memory usage for samples = 2**8")
plt.xlabel("number of samples")
plt.ylabel("memory")
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('memoryVaringClusters.png')
plt.clf()
plt.scatter(numSamples, times,color="red")
plt.title("VLAD: time usage for samples = 2**8")
plt.xlabel("number of samples")
plt.ylabel("time")
plt.savefig('timeVaringClusters.png')
print "Script ended. Results:"
print "numSamples: ",numSamples
print "vladMemory: ",memoris
print "vladTimes: ",times

import numpy as np

def gaussianData(samplesPerGaussian, dim):
    '''Generate numberOfClusters Gaussians samples with the same covariance matrix'''
    np.random.seed(312003)
    numberOfClusters=3
    C = np.random.randn(dim,dim)
    for cluster in range(numberOfClusters):
        X=np.dot(np.random.randn(samplesPerGaussian, dim), C)+cluster*np.random.randn(dim)
        print np.dot(np.random.randn(samplesPerGaussian, dim), C)+cluster*np.random.randn(dim)

gaussianData(100,32)

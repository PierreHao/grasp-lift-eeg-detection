This directory contains the onlineminibatchkmeans comparision run.

The files 
* onlineKmeansComparison2.py
* comparison.py
* kmeansPPComparison.py

correspond to the scripts used for running the performance tests.

The director plots2, plots, plot3 found one level above this directory contains various plots that were generated during the test run using 
NYU HPC clusters.

We ran the script using 64gb of RAM in hpc clusters.  The scripts were structured such that they can run in background for various input sizes and dimensions.

We were able to get a rule of thumb explaining the switchover for values of k and n for which to use onlinekmeans.
For `k > 8 && n > 2^10` we should switch over to onlinekmeansminibatch algorithm

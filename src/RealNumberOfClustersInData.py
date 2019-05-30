from matplotlib import pyplot as plt
from sklearn.manifold import MDS
from gap_statistic import OptimalK
import numpy as np
from DataGenerator import generateOneClusterData
from Settings import (DEFAULT_NUMBER_OF_FEATURES,
                        DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                        DEFAULT_FEATURE_MEAN_RANGE)

nDatasets = 20
clusterCounts = np.empty((nDatasets,))
randomNumberSeeds = list(range(nDatasets))

for i, randomNumberSeed in enumerate(randomNumberSeeds):

    data = generateOneClusterData(DEFAULT_NUMBER_OF_FEATURES,
                                DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                                DEFAULT_FEATURE_MEAN_RANGE,
                                randomNumberSeed,
                                distribution="normal")

    optimalK = OptimalK(parallel_backend='joblib', n_jobs=-1)
    clusterCount = optimalK(data, n_refs=3, cluster_array=np.arange(1, 10))
    clusterCounts[i] = clusterCount

plt.figure()
plt.title("Cluster Counts Found By Gap Statistic")
plt.xlabel("Data Set Id")
plt.ylabel("Number Of Clusters")
plt.bar(range(nDatasets), clusterCounts)
plt.show()

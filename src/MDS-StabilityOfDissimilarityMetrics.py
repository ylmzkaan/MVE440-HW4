import numpy as np
import os
from gap_statistic import OptimalK
from matplotlib import pyplot as plt
from sklearn.manifold import MDS
from sklearn.neighbors import DistanceMetric

from DataGenerator import generateOneClusterData
from Preprocessing import optimalClusterCount
from Settings import (DEFAULT_FEATURE_MEAN_RANGE, DEFAULT_NUMBER_OF_FEATURES,
                      DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                      DEFAULT_RANDOM_NUMBER_SEED)

metrics = ("euclidean", "manhattan", "chebyshev",
            "minkowski")
nMetrics = len(metrics)            
meanClusterCount = np.empty((nMetrics,))
stdClusterCount = np.empty((nMetrics,))

for j, metric in enumerate(metrics):

    nDifferentDataSet = 50
    clusterCounts = np.empty((nDifferentDataSet,))

    mds = MDS(n_components=6, n_jobs=-1, dissimilarity="precomputed")
    dist = DistanceMetric.get_metric(metric)
    print("MDS Metric: {}".format(metric))

    for i in range(nDifferentDataSet):

        data = generateOneClusterData(DEFAULT_NUMBER_OF_FEATURES,
                                    DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                                    DEFAULT_FEATURE_MEAN_RANGE, 
                                    i,
                                    distribution="normal")
        precomputedMetricData = dist.pairwise(data)                                    

        mdsData = mds.fit_transform(precomputedMetricData)

        optimalK = OptimalK(parallel_backend='joblib', n_jobs=-1)
        clusterCount = optimalK(mdsData, n_refs=3, cluster_array=np.arange(1, 10))
        clusterCounts[i] = clusterCount
    
    meanClusterCount[j] = np.mean(clusterCounts)
    stdClusterCount[j] = np.std(clusterCounts)

color = np.random.rand(1,3)
OPACITY = 0.7

plt.figure()        
plt.title("Metrics VS Cluster Counts")
plt.xlabel("Metric")
plt.ylabel("Cluster Counts")
plt.errorbar(metrics, meanClusterCount, yerr=stdClusterCount,
            capthick=2, capsize=10, linewidth=3)                 
plt.show()       

dirPath = os.path.dirname(os.path.realpath(__file__))
saveDir = os.path.dirname(dirPath)
filename = "MetricsVSClusterCounts.png"
plt.savefig(os.path.join(saveDir, filename))

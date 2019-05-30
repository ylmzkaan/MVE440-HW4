import numpy as np
import os
from gap_statistic import OptimalK
from matplotlib import pyplot as plt
from sklearn.manifold import MDS
from sklearn.neighbors import DistanceMetric

from DataGenerator import generateOneClusterData
from Settings import (DEFAULT_FEATURE_MEAN_RANGE, DEFAULT_NUMBER_OF_FEATURES,
                      DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                      DEFAULT_RANDOM_NUMBER_SEED)

metrics = ("euclidean", "manhattan", "chebyshev",
            "minkowski")
nMetrics = len(metrics)
meanStress = np.empty((nMetrics,))
stdStress = np.empty((nMetrics,))
meanClusterCount = np.empty((nMetrics,))
stdClusterCount = np.empty((nMetrics,))

nDifferentDataSet = 20
stress = np.empty((nDifferentDataSet,4))

for j, metric in enumerate(metrics):

    clusterCounts = np.empty((nDifferentDataSet,))

    dist = DistanceMetric.get_metric(metric)
    print("MDS Metric: {}".format(metric))

    for i in range(nDifferentDataSet):

        data = generateOneClusterData(DEFAULT_NUMBER_OF_FEATURES,
                                    DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                                    DEFAULT_FEATURE_MEAN_RANGE,
                                    i,
                                    distribution="normal")
        precomputedMetricData = dist.pairwise(data)

        mds = MDS(n_components=8, n_jobs=-1, dissimilarity="precomputed")
        mdsData = mds.fit_transform(precomputedMetricData)

        optimalK = OptimalK(parallel_backend='joblib', n_jobs=-1)
        clusterCount = optimalK(mdsData, n_refs=3, cluster_array=np.arange(1, 10))
        clusterCounts[i] = clusterCount
        stress[i,j] = mds.stress_

    meanClusterCount[j] = np.mean(clusterCounts)
    stdClusterCount[j] = np.std(clusterCounts)

    meanStress[j] = np.mean(stress[:,j])
    stdStress[j] = np.std(stress[:,j])

saveDir = os.path.join("data", "MDS-stressPerMetric.npy")
np.save(saveDir, stress)

plt.subplot(1,2,1)
plt.tight_layout()
plt.title("Metrics VS Cluster Counts")
plt.xlabel("Metric")
plt.ylabel("Cluster Counts")
plt.errorbar(metrics, meanClusterCount, yerr=stdClusterCount,
            capthick=2, capsize=10, linewidth=3)
plt.subplot(1,2,2)
plt.title("Metrics VS Cluster Counts")
plt.xlabel("Metric")
plt.ylabel("Stress")
plt.errorbar(metrics, meanStress, yerr=stdStress,
            capthick=2, capsize=10, linewidth=3)
plt.show()

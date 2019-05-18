import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import DistanceMetric
from sklearn.manifold import MDS

from DataGenerator import generateOneClusterData
from Settings import (DEFAULT_FEATURE_MEAN_RANGE, DEFAULT_NUMBER_OF_FEATURES,
                      DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                      DEFAULT_RANDOM_NUMBER_SEED)

metrics = ("euclidean", "manhattan", "chebyshev", "minkowski")

data = generateOneClusterData(DEFAULT_NUMBER_OF_FEATURES,
                            DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                            DEFAULT_FEATURE_MEAN_RANGE, 
                            DEFAULT_RANDOM_NUMBER_SEED,
                            distribution="uniform")

for i, metric in enumerate(metrics):
    dist = DistanceMetric.get_metric(metric)
    precomputedMetricData = dist.pairwise(data)
    mds = MDS(n_components=2, n_jobs=-1, dissimilarity="precomputed")
    mdsData = mds.fit_transform(precomputedMetricData)                            

    OPACITY = 0.7
    plt.subplot(2,2,i+1)
    plt.title("Data Set - {}".format(metric))
    plt.xlabel("MDS Feature 1")
    plt.ylabel("MDS Feature 2")
    plt.scatter(mdsData[:,0], mdsData[:,1],
                c=np.random.rand(1,3), alpha=OPACITY)
plt.show()                

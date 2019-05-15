from matplotlib import pyplot as plt
from sklearn.manifold import MDS
from Preprocessing import optimalClusterCount
from gap_statistic import OptimalK
import numpy as np
from DataGenerator import generateOneClusterData
from Settings import (DEFAULT_NUMBER_OF_FEATURES,
                        DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                        DEFAULT_FEATURE_MEAN_RANGE,
                        DEFAULT_RANDOM_NUMBER_SEED)

mdsNumberOfComponentsRange = range(1, 11)
meanClusterCount = np.empty((10,))
stdClusterCount = np.empty((10,))

for j, mdsNumberOfComponents in enumerate(mdsNumberOfComponentsRange):

    nClusterCounts = 20
    clusterCounts = np.empty((nClusterCounts,))
    print("MDS Number of Components: {}".format(mdsNumberOfComponents))

    for i in range(nClusterCounts):

        data = generateOneClusterData(DEFAULT_NUMBER_OF_FEATURES,
                                    DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                                    DEFAULT_FEATURE_MEAN_RANGE, 
                                    i,
                                    distribution="normal")

        mds = MDS(n_components=2, n_jobs=-1)
        mdsData = mds.fit_transform(data)

        optimalK = OptimalK(parallel_backend='joblib', n_jobs=-1)
        clusterCount = optimalK(mdsData, n_refs=3, cluster_array=np.arange(1, 10))
        clusterCounts[i] = clusterCount
    
    meanClusterCount[j] = np.mean(clusterCounts)
    stdClusterCount[j] = np.std(clusterCounts)

color = np.random.rand(1,3)
OPACITY = 0.7

plt.figure()        
plt.title("Data Sets With MDS")
plt.xlabel("MDS - Number Of Components")
plt.ylabel("Cluster Count")
plt.errorbar(mdsNumberOfComponentsRange, meanClusterCount, yerr=stdClusterCount,
            capthick=2, capsize=10)    
plt.show()                
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

clusterCounts = np.empty(20,)

for i in range(20):

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
    print("Cluster count: {}".format(clusterCount))

color = np.random.rand(1,3)
OPACITY = 0.7

plt.subplot(1, 3, 1)
plt.title("Data Set")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.scatter(data[:,0], data[:,1],
            c=color, alpha=OPACITY)

plt.subplot(1, 3, 2)
plt.title("Data Set With MDS")
plt.xlabel("MDS Feature 1")
plt.ylabel("MDS Feature 2")
plt.scatter(mdsData[:,0], mdsData[:,1],
            c=color, alpha=OPACITY)

plt.subplot(1, 3, 3)        
plt.title("Data Sets With MDS")
plt.xlabel("Data Set Idx")
plt.ylabel("Cluster Numbers")
plt.plot(range(20), clusterCounts)    
plt.show()                
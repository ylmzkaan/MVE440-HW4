import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import MDS

from DataGenerator import generateOneClusterData
from Preprocessing import optimalClusterCount
from Settings import (DEFAULT_FEATURE_MEAN_RANGE, DEFAULT_NUMBER_OF_FEATURES,
                      DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                      DEFAULT_RANDOM_NUMBER_SEED)

clusterCounts = np.empty(20,)

for i in range(20):

    data = generateOneClusterData(DEFAULT_NUMBER_OF_FEATURES,
                                DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                                DEFAULT_FEATURE_MEAN_RANGE, 
                                i,
                                distribution="uniform")

    mds = MDS(n_components=2, n_jobs=-1)
    mdsData = mds.fit_transform(data)

    clusterCounts[i] = optimalClusterCount(mdsData)
    print("Cluster count: {}".format(clusterCounts[i]))

plt.figure()
plt.title("Data Sets With MDS")
plt.xlabel("Data Set Idx")
plt.ylabel("Cluster Numbers")
plt.plot(range(20), clusterCounts)
plt.show()

color = np.random.rand(1,3)
OPACITY = 0.7

plt.subplot(1, 2, 1)
plt.title("Data Set With MDS")
plt.xlabel("MDS Feature 1")
plt.ylabel("MDS Feature 2")
plt.scatter(mdsData[:,0], mdsData[:,1],
            c=color, alpha=OPACITY)

plt.subplot(1, 2, 2)
plt.title("Data Set")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.scatter(data[:,0], data[:,1],
            c=color, alpha=OPACITY)
plt.show()                

from matplotlib import pyplot as plt
from sklearn.manifold import MDS
from gap_statistic import OptimalK
import numpy as np
import os
from DataGenerator import generateOneClusterData
from Settings import (DEFAULT_NUMBER_OF_FEATURES,
                        DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                        DEFAULT_FEATURE_MEAN_RANGE,
                        DEFAULT_RANDOM_NUMBER_SEED)

nDifferentMdsComponentNumber = 20
mdsNumberOfComponentsRange = range(1, 1+nDifferentMdsComponentNumber)
meanStress = np.empty((nDifferentMdsComponentNumber,))
stdStress = np.empty((nDifferentMdsComponentNumber,))

meanClusterCount = np.empty((nDifferentMdsComponentNumber,))
stdClusterCount = np.empty((nDifferentMdsComponentNumber,))

nDatasets = 20
stress = np.empty((nDatasets, nDifferentMdsComponentNumber))

for j, mdsNumberOfComponents in enumerate(mdsNumberOfComponentsRange):

    clusterCounts = np.empty((nDatasets,))

    print("MDS Number of Components: {}".format(mdsNumberOfComponents))

    for iDataset in range(nDatasets):

        data = generateOneClusterData(DEFAULT_NUMBER_OF_FEATURES,
                                    DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                                    DEFAULT_FEATURE_MEAN_RANGE,
                                    iDataset,
                                    distribution="normal")

        mds = MDS(n_components=mdsNumberOfComponents, n_jobs=-1)
        mdsData = mds.fit_transform(data)
        stress[iDataset,j] = mds.stress_

        optimalK = OptimalK(parallel_backend='joblib', n_jobs=-1)
        clusterCount = optimalK(mdsData, n_refs=3, cluster_array=np.arange(1, 10))
        clusterCounts[iDataset] = clusterCount

    meanClusterCount[j] = np.mean(clusterCounts)
    stdClusterCount[j] = np.std(clusterCounts)

    meanStress[j] = np.mean(stress[:,j])
    stdStress[j] = np.std(stress[:,j])

saveDir = os.path.join("data", "MDS-meanStressForDifferentComponentNumber.npy")
np.save(saveDir, stress)

plt.subplot(1,2,1)
plt.title("MDS - Number Of Components VS Cluster Count")
plt.xlabel("MDS - Number Of Components")
plt.ylabel("Cluster Count")
plt.errorbar(mdsNumberOfComponentsRange, meanClusterCount, yerr=stdClusterCount,
            capthick=2, capsize=5, linewidth=3, label="MDS Data")
plt.xticks(mdsNumberOfComponentsRange, mdsNumberOfComponentsRange)

plt.subplot(1,2,2)
plt.title("MDS - Number Of Components VS Stress")
plt.xlabel("MDS - Number Of Components")
plt.ylabel("Stress")
plt.errorbar(mdsNumberOfComponentsRange, meanStress, yerr=stdStress,
            capthick=2, capsize=5, linewidth=3)
plt.xticks(mdsNumberOfComponentsRange, mdsNumberOfComponentsRange)
plt.show()

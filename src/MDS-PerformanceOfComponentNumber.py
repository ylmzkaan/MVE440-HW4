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
meanStress = np.empty((10,))
stdStress = np.empty((10,))

meanClusterCountOrgData = np.empty((10,))
stdClusterCountOrgData = np.empty((10,))

meanClusterCount = np.empty((10,))
stdClusterCount = np.empty((10,))

for j, mdsNumberOfComponents in enumerate(mdsNumberOfComponentsRange):

    nClusterCounts = 50
    clusterCounts = np.empty((nClusterCounts,))
    clusterCountsOrgData = np.empty((nClusterCounts,))
    stress = np.empty((nClusterCounts,))

    print("MDS Number of Components: {}".format(mdsNumberOfComponents))

    for i in range(nClusterCounts):

        data = generateOneClusterData(DEFAULT_NUMBER_OF_FEATURES,
                                    DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                                    DEFAULT_FEATURE_MEAN_RANGE, 
                                    i,
                                    distribution="normal")

        mds = MDS(n_components=mdsNumberOfComponents, n_jobs=-1)
        mdsData = mds.fit_transform(data)
        stress[i] = mds.stress_

        optimalK = OptimalK(parallel_backend='joblib', n_jobs=-1)
        clusterCount = optimalK(mdsData, n_refs=3, cluster_array=np.arange(1, 10))
        clusterCounts[i] = clusterCount
        clusterCount = optimalK(data, n_refs=3, cluster_array=np.arange(1, 10))
        clusterCountsOrgData[i] = clusterCount
    
    meanClusterCount[j] = np.mean(clusterCounts)
    stdClusterCount[j] = np.std(clusterCounts)

    meanClusterCountOrgData[j] = np.mean(clusterCountsOrgData)
    stdClusterCountOrgData[j] = np.std(clusterCountsOrgData)

    meanStress[j] = np.mean(stress)
    stdStress[j] = np.std(stress)


plt.subplot(1,2,1)        
plt.title("MDS - Number Of Components VS Cluster Count")
plt.xlabel("MDS - Number Of Components")
plt.ylabel("Cluster Count")
plt.errorbar(mdsNumberOfComponentsRange, meanClusterCount, yerr=stdClusterCount,
            capthick=2, capsize=10, linewidth=3, label="MDS Data")
plt.errorbar(mdsNumberOfComponentsRange, meanClusterCountOrgData, yerr=stdClusterCountOrgData,
            capthick=2, capsize=10, linewidth=3, label="Original Data")            
plt.legend()            
plt.subplot(1,2,2)        
plt.title("MDS - Number Of Components VS Stress")
plt.xlabel("MDS - Number Of Components")
plt.ylabel("Stress")
plt.errorbar(mdsNumberOfComponentsRange, meanStress, yerr=stdStress,
            capthick=2, capsize=10, linewidth=3)                             
plt.show()                
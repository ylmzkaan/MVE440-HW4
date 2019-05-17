from matplotlib import pyplot as plt
from sklearn.manifold import MDS
from gap_statistic import OptimalK
import numpy as np
from sklearn.neighbors import DistanceMetric

from DataGenerator import generateOneClusterData
from Settings import (DEFAULT_NUMBER_OF_FEATURES,
                        DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                        DEFAULT_FEATURE_MEAN_RANGE,
                        DEFAULT_RANDOM_NUMBER_SEED)

defaultStressPerMDSComponentNumber = [964234,149740520,41874,964234]
nPerturbations = 20
nDataSets = 10
metrics = ("euclidean", "manhattan", "chebyshev",
            "minkowski")
meanStressDiff = np.empty((4,nDataSets))
stdStressDiff = np.empty((4,nDataSets))

for j, metric in enumerate(metrics):
    dist = DistanceMetric.get_metric(metric)
    for k in range(nDataSets):
        sampleIdxToDelete = np.random.choice(DEFAULT_NUMBER_OF_RECORDS_PER_CLASS, nPerturbations, replace=False)
        stressDiff = np.empty((nPerturbations,))

        print("MDS Metric: {}".format(metric))

        data = generateOneClusterData(DEFAULT_NUMBER_OF_FEATURES,
                                        DEFAULT_NUMBER_OF_RECORDS_PER_CLASS,
                                        DEFAULT_FEATURE_MEAN_RANGE, 
                                        k,
                                        distribution="normal")                                                             

        for i in range(nPerturbations):

            #dataToProcess = data.copy()
            #dataToProcess[sampleIdxToDelete[i]] *= 1.2
            dataToProcess = np.delete(data, sampleIdxToDelete[i], axis=0)
            precomputedMetricData = dist.pairwise(dataToProcess)

            mds = MDS(n_components=7, n_jobs=-1, dissimilarity="precomputed")
            mdsData = mds.fit_transform(precomputedMetricData) 
            stressDiff[i] = defaultStressPerMDSComponentNumber[j] - mds.stress_

        meanStressDiff[j,k] = np.mean(stressDiff)
        stdStressDiff[j,k] = np.std(stressDiff)

plt.figure()
for i in range(10):
    plt.errorbar(metrics, meanStressDiff[:,i], yerr=stdStressDiff[:,i],
                capthick=2, capsize=10, linewidth=3, label="Data Set: {}".format(i))  
plt.xlabel("Metrics")
plt.ylabel("Stress Deviation After Small Perturbation")
plt.title("Mean Stress Deviation After Perturbation")
plt.legend()
plt.show()
